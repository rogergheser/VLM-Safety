from pathlib import Path
import lightning as L
import torch
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_module import LLavaDataset
from utils.types import PreProcessedModelInput
from utils.utils import (
    _get_default_safe_lora_config,
    load_model,
    find_all_linear_names,
    train_collate_fn,
    eval_collate_fn,
    get_expected_image_size,
)
from transformers import LlavaForConditionalGeneration, LlavaProcessor # type: ignore
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from utils.metrics import Metrics
from typing import Union

@dataclass(eq=False)
class My_LLava(L.LightningModule):
    model_path: str = "liuhaotian/llava-v1.5-7b"
    use_lora: bool = True
    use_qlora: bool = False
    safe_lora: bool = False
    dataset_name: str = "aimagelab/ViSU-Text"
    batch_size: int = 8
    num_workers: int = 4
    MAX_LENGTH: int = 64
    metrics : Metrics = field(default_factory=Metrics)
    config : dict = field(default_factory=dict)
    train_set: LLavaDataset = field(init=False)
    val_set: LLavaDataset = field(init=False)
    test_set: LLavaDataset = field(init=False)
    lora_config: LoraConfig = field(init=False)
    model : Union[PeftModel, PeftMixedModel] = field(init=False)
    raw_model : LlavaForConditionalGeneration = field(init=False)
    processor : LlavaProcessor = field(init=False)
    image_size: tuple[int, int] = field(default=(224, 224), init=False)
    unsafe_percent: float = 0.2

    @staticmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
    ) -> PeftModel:
        """
        Create a peft My_LLava instance from a checkpoint.
        """
        peft_state_dict = torch.load(checkpoint_path, map_location="cpu").half()
        model = PeftModel.from_pretrained(
            LlavaForConditionalGeneration.from_pretrained(
                peft_state_dict['model_path'],
                device_map="auto",
            ),
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        return model

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "My_LLava":
        """
        Create a My_LLava instance from a config dictionary.
        """
        return cls(
            model_path=config.get("model_path", "liuhaotian/llava-v1.5-7b"),
            use_lora=config.get("use_lora", False),
            use_qlora=config.get("use_qlora", False),
            safe_lora=config.get("safe_lora", False),
            dataset_name=config.get("dataset_name", "aimagelab/ViSU-Text"),
            batch_size=config.get("batch_size", 1),
            num_workers=config.get("num_workers", 4),
            MAX_LENGTH=config.get("MAX_LENGTH", 64),
            unsafe_percent=config.get("unsafe_percent", 0.2),
            config=config,
        )

    def __post_init__(
        self,
    ):  
        super().__init__()
        self.processor, self.raw_model = load_model(
            model_name=self.model_path,
            use_lora=self.use_lora,
            use_qlora=self.use_qlora,
        )
        self.processor.tokenizer.padding_side = "right" # type: ignore
        assert self.raw_model is not None, "Model is None after loading"
        self.image_size = get_expected_image_size(self.raw_model)

        self.prepare_dataset()
        
        if self.use_qlora:
            prepare_model_for_kbit_training(self.raw_model)
        assert self.raw_model is not None, "Model is None after kbit training preparation"
        print("Model type: ", type(self.raw_model))
        self.model = get_peft_model(
            self.raw_model, self._get_lora_config(),
            autocast_adapter_dtype=False
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.model.print_trainable_parameters()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return self.model._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return self.model._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        model_device = next(self.model.parameters()).device
        if isinstance(batch, PreProcessedModelInput):
            input_ids, attention_mask, pixel_values, labels, dict_labels = batch.deconstruct()
            return PreProcessedModelInput(
                input_ids=input_ids.to(model_device, non_blocking=True),
                attention_mask=attention_mask.to(model_device, non_blocking=True),
                pixel_values=pixel_values.to(model_device, non_blocking=True),
                labels=labels.to(model_device, non_blocking=True),
                dict_labels=dict_labels,
            )
        return super().transfer_batch_to_device(batch, model_device, dataloader_idx)

    def training_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels, _ = batch.deconstruct()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels, # type: ignore
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels, labels_dict = batch.deconstruct()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens= self.MAX_LENGTH
            )
                             
        predictions: list[str] = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        print(type(predictions))
        print(type(predictions[0]))
        for pred, captions in zip(predictions, labels):
            self.metrics.compute(pred, captions) # type: ignore
        
        average_scores = self.metrics.average_scores
        # self.log("val_bleu", average_scores["bleu"])
        self.log("val_rouge", average_scores["rouge"])

        return torch.tensor(average_scores["rouge"])
    
    def test_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels, _ = batch.deconstruct()
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=self.MAX_LENGTH
        )
        predictions: list[str] = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        print(type(predictions))
        print(type(predictions[0]))
        for pred, label in zip(predictions, labels['nsfw']):
            self.metrics.compute(pred, label)
        average_scores = self.metrics.average_scores
        # self.log("test_bleu", average_scores["bleu"])
        self.log("test_rouge", average_scores["rouge"])
        return torch.tensor(average_scores["rouge"])

    def configure_optimizers(self)-> torch.optim.Optimizer:
        """Returns a default AdamW optimizer"""
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 1e-4))

        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_set,
            batch_size=self.batch_size,
            collate_fn=partial(train_collate_fn, processor=self.processor, prob_unsafe=self.unsafe_percent), # type: ignore
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_set,
            batch_size=self.batch_size,
            collate_fn=partial(train_collate_fn, processor=self.processor, prob_unsafe=self.unsafe_percent), # type: ignore
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.test_set,
            batch_size=self.batch_size,
            collate_fn=partial(eval_collate_fn, processor=self.processor), # type: ignore
            num_workers=self.num_workers,
        )

    def prepare_dataset(self):
        """
        Prepare the dataset for training and validation.
        """
        self.train_set, self.val_set, self.test_set = LLavaDataset.splits_from_name(
            dataset_name=self.dataset_name,
            splits=(0.8, 0.1, 0.1),
            size=self.image_size
        )
        print(f"Dataset {self.dataset_name} loaded with {len(self.train_set)} train samples, {len(self.val_set)} val samples, and {len(self.test_set)} test samples.")

    def _get_lora_config(self):
        lora_config = LoraConfig(
            r=2,
            lora_alpha=32,
            target_modules=find_all_linear_names(self.raw_model),
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return lora_config