import logging
import wandb
from typing import Any

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def log_captions_and_gts(test_values: list[dict[str, Any]]):
    if not test_values:
        print("No test values found for logging.")
        return

    table = wandb.Table(columns=["Index", "Prediction", "Ground Truth", "Toxic Caption"])

    for i, sample in enumerate(test_values):
        table.add_data(
            i,
            sample["pred"],
            sample["safe"],
            sample["nsfw"],
        )
    
    wandb.log({"captions_vs_ground_truths": table})