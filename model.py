from functools import partial
import lightning as L
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from data_module import LLavaDataset
from utils import (
    load_model,
    find_all_linear_names,
    train_collate_fn,
    eval_collate_fn,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

@dataclass
class My_LLava(L.LightningModule):
    model_path: str = "liuhaotian/llava-v1.5-7b"
    lora_config: LoraConfig | None = None
    use_lora: bool = False
    use_qlora: bool = False
    dataset_name: str = "aimagelab/ViSU-Text"
    batch_size: int = 8
    num_workers: int = 4

    def __post_init__(
        self,
    ):  
        self.processor, model = load_model(
            model_name=self.model_path,
            use_lora=self.use_lora,
            use_qlora=self.use_qlora,
        )
        self.processor.tokenizer.padding_side = "right"

        if self.lora_config is None:
            self.lora_config = self._get_lora_config()

        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, self.lora_config)
        self.model.print_trainable_parameters()
    
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
            r=16,
            lora_alpha=32,
            target_modules=find_all_linear_names(self.model),
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return lora_config