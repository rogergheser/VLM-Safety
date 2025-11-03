from pathlib import Path
import lightning as L
import torch
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_module import LLavaDataset
from utils.types import PreProcessedModelInput
from utils.utils import (
    dict_list_to_list_dict,
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
    unsafe_percent: float = 0.2
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

    def _filter_state_dict(self, state_dict):
        """Filter the state dict to only include LoRA parameters."""
        avoid = ["raw_model", "vision_tower"]
        allowed = ["k_proj", "v_proj"]
        lora_state_dict = {
            k: v for k, v in state_dict.items()
            if "lora" in k and all(n not in k for n in avoid) and any(n in k for n in allowed) and "language_model" in k
        }
        return lora_state_dict

    def on_load_checkpoint(self, checkpoint):
        lora_state_dict = checkpoint.get("lora_state_dict", None)
        if lora_state_dict is None:
            print("⚠️  No LoRA adapters found in checkpoint.")
            return
        missing, unexpected = self.model.load_state_dict(lora_state_dict, strict=False)
        print(f"✅ Loaded LoRA adapters. Missing={len(missing)}, Unexpected={len(unexpected)}")

    def on_save_checkpoint(self, checkpoint):
        full_state_dict = self.model.state_dict()
        lora_state_dict = {k: v.cpu() for k, v in full_state_dict.items() if "lora" in k}
        checkpoint["lora_state_dict"] = lora_state_dict
        checkpoint["lightning_module_init"] = {
            "model_path": self.model_path,
            "use_lora": self.use_lora,
            "use_qlora": self.use_qlora,
            "safe_lora": self.safe_lora,
            "dataset_name": self.dataset_name,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "unsafe_percent": self.unsafe_percent,
            "MAX_LENGTH": self.MAX_LENGTH,
        }
        print(f"Saved LoRA-only state dict with {len(lora_state_dict)} keys.")

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

        self.model = get_peft_model(
            self.raw_model, self._get_lora_config(),
            autocast_adapter_dtype=False
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.model.print_trainable_parameters()

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
        if (torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any() or
            torch.isnan(input_ids).any() or torch.isinf(input_ids).any()):
            print(f"NaN/Inf detected in batch {batch_idx}")
        loss = outputs.loss
        if (torch.isnan(loss)).any():
            breakpoint()
            print("Hit a nan in loss")
            print(loss)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels, labels_dict = batch.deconstruct()
        labels_dict = dict_list_to_list_dict(labels_dict)
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

        for pred, captions in zip(predictions, labels_dict):
            self.metrics.compute(pred, captions) # type: ignore
        
        average_scores = self.metrics.average_scores
        # self.log("val_bleu", average_scores["bleu"])
        self.log("val_rouge", average_scores["rouge"])
        self.log("rouge-utility", average_scores["rouge-utility"])
        self.log("rouge-safety", average_scores["rouge-safety"])
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 1e-5), eps=1e-6)

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