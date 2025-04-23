import lightning.pytorch as L
from pprint import pprint
from data_module import LLavaDataset
from model import My_LLava
from utils.utils import *
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

USE_LORA = False
USE_QLORA = True
MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO_ID = "rogergheser/llava-finetuning"
WANDB_PROJECT = "LLaVa"
WANDB_NAME = "llava-safe-nsfw"

if __name__ == '__main__': 
    config = {
        "model_path": MODEL_ID,
        "use_lora": USE_LORA,
        "use_qlora": USE_QLORA,
        "dataset_name": "aimagelab/ViSU-Text",
        "num_workers": 4,
        "max_epochs": 10,
        "MAX_LENGTH": 64,
        # "val_check_interval": 0.2, # how many times we want to validate during an epoch
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 2,
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }   
    if torch.cuda.is_available():
        print("Using GPU\n")
    else:
        print("Using CPU\n")
    pprint(config)
    
    train_dataset = LLavaDataset("aimagelab/ViSU-Text", split="test")

    model_module = My_LLava.from_config(config)
    early_stop_callback = EarlyStopping(
        monitor="rouge",
        patience=3,
        verbose=True,
        mode="max",
    )

    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

    trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=config.get("max_epochs"),
            accumulate_grad_batches=config.get("accumulate_grad_batches"),
            check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
            gradient_clip_val=config.get("gradient_clip_val"),
            precision="16-mixed",
            limit_val_batches=5,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[early_stop_callback],
    )

    trainer.fit(model_module)
