import lightning as L
import torch
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_module import LLavaDataset
from SafeLoRA.model import SafeLoRA
from SafeLoRA.config import SafeLoRAConfig
from utils.llava_dtypes import PreProcessedModelInput
from utils.log import log_captions_and_gts
from utils.utils import (
    dict_list_to_list_dict,
    load_model,
    find_all_linear_names,
    llava_collate_fn,
    get_expected_image_size,
)
from transformers import LlavaForConditionalGeneration, LlavaProcessor # type: ignore
from peft import (
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    get_peft_model,
)
from utils.metrics import Metrics, TestMetrics
from typing import Any, Union
import os

@dataclass(eq=False)
class My_LLava(L.LightningModule):
    model_path: str = "liuhaotian/llava-v1.5-7b"
    use_lora: bool = True
    use_qlora: bool = False
    safe_lora: bool = False
    dataset_name: str = "aimagelab/ViSU-Text"
    batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8
    num_workers: int = 4
    unsafe_percent: float = 0.2
    MAX_LENGTH: int = 64
    metrics : Metrics = field(default_factory=Metrics)
    test_metrics: TestMetrics = field(default_factory=TestMetrics)
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
            val_batch_size=config.get("val_batch_size", 1),
            test_batch_size=config.get("test_batch_size", 1),
            num_workers=config.get("num_workers", 4),
            MAX_LENGTH=config.get("MAX_LENGTH", 64),
            unsafe_percent=config.get("unsafe_percent", 0.2),
            config=config,
        )

    # def on_load_checkpoint(self, checkpoint: dict) -> None:
    #     print("Loaded pretrained peft model")
    #     self.processor = LlavaProcessor.from_pretrained(self.model_path)
    #     self.model = PeftModel.from_pretrained(
    #         self.raw_model,
    #         "clean_ckp/peft_model",
    #     )

    def load_state_dict(self, state_dict, strict: bool = True):
        print("⚠️ Skipping default Lightning state_dict loading (handled manually).")
        return  # do nothing

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        print("Loaded pretrained peft model")
        del self.model
        self.processor = LlavaProcessor.from_pretrained(self.model_path)

        # load base model
        # base_model = LlavaForConditionalGeneration.from_pretrained(self.model_path)

        # re-wrap with same LoRA config
        peft_config = PeftConfig.from_pretrained("clean_ckp/peft_model")
        # self.model = get_peft_model(base_model, peft_config)
        # breakpoint()
        # # load only adapter weights
        # self.model.load_adapter("clean_ckp/peft_model", adapter_name="default")
        # print("Loaded LoRA adapter successfully ✅")

        self.model = PeftModel.from_pretrained(
            self.raw_model,
            "clean_ckp/peft_model",
            config=peft_config,
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        os.makedirs("clean_ckp", exist_ok=True)
        self.model.save_pretrained("clean_ckp/peft_model")

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
        self.image_size = get_expected_image_size(self.raw_model)

        self.train_set, self.val_set, self.test_set = LLavaDataset.splits_from_name(
            dataset_name=self.dataset_name,
            splits=(0.8, 0.1, 0.1),
            size=self.image_size
        )

        self.model = get_peft_model(
            self.raw_model, self._get_lora_config(),
            autocast_adapter_dtype=False
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.model.print_trainable_parameters()

    def training_step(
        self,
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
            print("Hit a nan in loss")
            print(loss)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(
        self,
        batch: PreProcessedModelInput,
        batch_idx: int,
    ) -> None:
        input_ids, attention_mask, pixel_values, _, labels_dict = batch.deconstruct()
        labels_dict = dict_list_to_list_dict(labels_dict)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens= self.MAX_LENGTH
            )
                             
        predictions: list[str] = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        self.metrics.compute(predictions, labels_dict)
    
    def on_validation_epoch_end(self) -> None:
        average_scores = self.metrics.average_scores
        for key, value in average_scores.items():
            self.log(f"val_{key}", value)

    def test_step(self,
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

        self.test_metrics.update(predictions, labels_dict)
        return torch.tensor([0.0])
        
    def on_test_end(self) -> None:

        self.test_metrics.compute_all()
        average_scores = self.test_metrics.average_scores
        for key, value in average_scores.items():
            self.logger.experiment.log({f"test_{key}": value})

        log_captions_and_gts(self.test_metrics.values)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Returns a default AdamW optimizer"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 1e-6), eps=1e-6)

        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_set,
            batch_size=self.batch_size,
            collate_fn=partial(llava_collate_fn, processor=self.processor), # type: ignore
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_set,
            batch_size=self.val_batch_size,
            collate_fn=partial(llava_collate_fn, processor=self.processor),
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.test_set,
            batch_size=self.test_batch_size,
            collate_fn=partial(llava_collate_fn, processor=self.processor),
            num_workers=self.num_workers,
        )

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

    def apply_safe_lora(self, aligned_model_path: str, unaligned_model_path: str):
        pmodel = self.model.language_model
        config = SafeLoRAConfig(
            base_model_path=unaligned_model_path,
            aligned_model_path=aligned_model_path,
            select_layers_type='threshold',
            threshold=0.5,
            devices=self.model.device,
        )
        peft_config = self.model.peft_config["default"]

        safelora = SafeLoRA(pmodel, config, peft_config)
        self.model.language_model = safelora.model
