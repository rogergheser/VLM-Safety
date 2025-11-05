import lightning.pytorch as L
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from pprint import pprint
from model import My_LLava
from utils.utils import *
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

USE_LORA = True
USE_QLORA = False
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
        "num_workers": 8,
        "max_epochs": 5,
        "MAX_LENGTH": 64,
        # "val_check_interval": 0.2, # how many times we want to validate during an epoch
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 0.1,
        "accumulate_grad_batches": 1,
        "lr": 1e-6,
        "batch_size": 1,
        "val_batch_size": 32,
        "test_batch_size": 32,
        "seed":1,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "unsafe_percentage": 0.2,
        "verbose": True,
        "debug": True,
    }   
    if torch.cuda.is_available():
        print("Using GPU\n")
    elif torch.backends.mps.is_available():
        print("Using MPS\n")
        config['device'] = 'mps'
    else:
        print("Using CPU\n")
    torch.set_float32_matmul_precision('high')
    torch.autograd.set_detect_anomaly(True)
    pprint(config)

    model_module = My_LLava.from_config(config)

    # eval_dataset = LLavaDataset(
    #     "aimagelab/ViSU-Text",
    #     split="test", 
    #     size=model_module.image_size
    # )

    early_stop_callback = EarlyStopping(
        monitor="rouge-safety",
        patience=1,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        every_n_epochs=1,
        save_last=True,
    )

    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)
    wandb_logger.log_hyperparams(config)
    torch.set_float32_matmul_precision("high")  # use A30 tensor cores

    trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            strategy="auto",
            num_nodes=1,
            max_epochs=config.get("max_epochs"),
            accumulate_grad_batches=config.get("accumulate_grad_batches", 8),
            check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
            gradient_clip_val=config.get("gradient_clip_val"),
            precision=32,
            limit_val_batches=5,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[
                # early_stop_callback,
                checkpoint_callback,
            ],
    )
    # trainer.validate(model_module, ckpt_path="last")
    trainer.fit(model_module, ckpt_path="last")

    trainer.test(model_module, ckpt_path="last")
