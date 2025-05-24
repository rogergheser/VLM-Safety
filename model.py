import os
import lightning as L
import torch
from functools import partial
from torch import LongTensor
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_module import LLavaDataset
from utils.types import PreProcessedModelInput
from utils.utils import (
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
    use_lora: bool = False
    use_qlora: bool = False
    safe_lora: bool = False
    dataset_name: str = "aimagelab/ViSU-Text"
    batch_size: int = 8
    num_workers: int = 4
    MAX_LENGTH: int = 64
    metrics : Metrics = field(default_factory=Metrics)
    config : dict = field(default_factory=dict)
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
            batch_size=config.get("batch_size", 8),
            num_workers=config.get("num_workers", 4),
            MAX_LENGTH=config.get("MAX_LENGTH", 64),
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

        prepare_model_for_kbit_training(self.raw_model)
        assert self.raw_model is not None, "Model is None after kbit training preparation"
        print("Model type: ", type(self.raw_model))
        self.model = get_peft_model(
            self.raw_model, self._get_lora_config()
        )
        self.model.print_trainable_parameters()

    def training_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels = batch.deconstruct()

        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            pixel_values=pixel_values.to(self.device),
            labels=labels.to(self.device), # type: ignore
            max_new_tokens=self.MAX_LENGTH,
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels = batch.deconstruct()
        
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens= self.MAX_LENGTH
        )
                             
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        scores = []
        for pred, label in zip(predictions, labels['nsfw']):
            self.metrics.compute(pred, label) # type: ignore
        
        average_scores = self.metrics.average_scores
        self.log("val_bleu", average_scores["bleu"])
        self.log("val_rouge", average_scores["rouge"])

        return torch.tensor(average_scores["rouge"])
    
    def configure_optimizers(self)-> torch.optim.Optimizer:
        """Returns a default AdamW optimizer"""
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 1e-4))

        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = LLavaDataset(self.dataset_name, split="test"),
            batch_size=self.batch_size,
            collate_fn=partial(train_collate_fn, processor=self.processor),
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = LLavaDataset(self.dataset_name, split="test"),
            batch_size=self.batch_size,
            collate_fn=partial(eval_collate_fn, processor=self.processor), # type: ignore
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