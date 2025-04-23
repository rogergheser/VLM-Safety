import lightning as L
import torch
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass
from data_module import LLavaDataset
from utils.types import PreProcessedModelInput
from utils.utils import (
    load_model,
    find_all_linear_names,
    train_collate_fn,
    eval_collate_fn,
)
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from utils.metrics import Metrics
from typing import Union

@dataclass
class My_LLava(L.LightningModule):
    model_path: str = "liuhaotian/llava-v1.5-7b"
    use_lora: bool = False
    use_qlora: bool = False
    safe_lora: bool = False
    dataset_name: str = "aimagelab/ViSU-Text"
    batch_size: int = 8
    num_workers: int = 4
    MAX_LENGTH: int = 64
    metrics : Metrics = Metrics()
    config : dict = None
    lora_config: LoraConfig = None
    model : Union[PeftModel, PeftMixedModel] = None

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
        self.processor.tokenizer.padding_side = "right"

        if self.lora_config is None:
            self.lora_config = self._get_lora_config()

        self.raw_model = prepare_model_for_kbit_training(self.raw_model)
        assert self.raw_model is not None, "Model is None after kbit training preparation"
        print("Model type: ", type(self.raw_model))
        self.raw_model.language_model = get_peft_model(
            self.raw_model.language_model, self.lora_config
        )
        print("Target modules:", find_all_linear_names(self.raw_model.language_model))
        print("LM type:", type(self.raw_model.language_model))

        assert self.raw_model.language_model is not None, "Language model is None after PEFT model preparation"
        self.model = self.raw_model  # now wrap the whole model again
        assert self.model is not None, "Model is None after PEFT model preparation"
        self.model.print_trainable_parameters()
    
    def training_step(self,
            batch: PreProcessedModelInput,
            batch_idx: int,
        ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values, labels = batch.deconstruct()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
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
        for pred, label in zip(predictions, labels):
            self.metrics.compute(pred, label)
        
        average_scores = self.metrics.average_scores
        self.log("val_bleu", average_scores["bleu"])
        self.log("val_rouge", average_scores["rouge"])

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            train_dataset = LLavaDataset(self.dataset_name, split="train"),
            batch_size=self.batch_size,
            collate_fn=partial(train_collate_fn, processor=self.processor),
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            val_dataset = LLavaDataset(self.dataset_name, split="test"),
            batch_size=self.batch_size,
            collate_fn=partial(eval_collate_fn, processor=self.processor),
            num_workers=self.num_workers,
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        return loss
    
    def _get_lora_config(self):
        lora_config = LoraConfig(
            r=2,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return lora_config