import torch
import pytorch_lightning as pl

# Hyperparameters for your LightningModule
hparams_dict = {
    "model_path": "llava-hf/llava-1.5-7b-h",
    "use_lora": True,
    "use_qlora": False,
    "safe_lora": True,
    "dataset_name": "aimagelab/ViSU-Text",
    "batch_size": 1,
    "num_workers": 4,
    "unsafe_percent": 0.2,
    "MAX_LENGTH": 64,
}

# 1. Load the old checkpoint
checkpoint = torch.load("checkpoints/last.ckpt", map_location="cpu")

# 2. Extract only LoRA parameters
lora_state_dict = {k: v.cpu() for k, v in checkpoint["state_dict"].items() if "lora" in k}

# 3. Create a Lightning-style checkpoint dict
lightning_ckpt = checkpoint | {
    "state_dict": lora_state_dict,                   # LoRA weights only
    "pytorch-lightning_version": "2.5.5",     # required by Lightning
    "lightning_module_init": hparams_dict,               # used to init your module
}

# 4. Save the Lightning-style checkpoint
torch.save(lightning_ckpt, "clean_ckp/last.ckpt")
print(f"✅ Saved Lightning-style LoRA-only checkpoint with {len(lora_state_dict)} parameters")
