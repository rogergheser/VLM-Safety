# repair_ckpt.py
import torch

IN_CKPT = "checkpoints/last.ckpt"   # <- your current ckpt path
OUT_CKPT = "checkpoints/last_filtered.ckpt"

ckpt = torch.load(IN_CKPT, map_location="cpu")
if "state_dict" in ckpt:
    state = ckpt["state_dict"]
    new_state = {}
    for k, v in state.items():
        breakpoint()
        # drop raw_model and processor keys (same filter as on_save_checkpoint)
        if k.startswith("raw_model.") or k.startswith("processor.") or k.startswith("model.base_model.model.") or "lora" not in k:
            continue
        breakpoint()
        new_state[k] = v
    ckpt["state_dict"] = new_state
else:
    print("No state_dict found in checkpoint — aborting.")

torch.save(ckpt, OUT_CKPT)
print("Saved filtered checkpoint to", OUT_CKPT)
