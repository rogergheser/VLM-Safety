import torch
from dataclasses import dataclass
from typing import Optional, Union
from datasets import load_dataset

@dataclass
class ModelInput:
    """
    ModelInput is a dataclass that represents the input to a model.
    It contains the following fields:
    - image: the path to the image
    - safe: the safe caption of the image
    - nsfw: the nsfw caption of the image
    """
    image: str
    safe: str
    nsfw: str

@dataclass
class PreProcessedModelInput:
    """
    PreProcessedModelInput is a dataclass that represents the preprocessed input to a model.
    It contains the following fields:
    - input_ids: the input ids of the image
    - attention_mask: the attention mask of the image
    - pixel_values: the pixel values of the image
    - labels: the labels of the image encoded for training and decoded for evaluation
    """
    input_ids: torch.LongTensor # Shape (batch_size, seq_len)
    attention_mask: torch.Tensor # Shape (batch_size, seq_len)
    pixel_values: torch.FloatTensor # Shape (batch_size, num_channels, height, width) 
    labels: torch.LongTensor | dict[str, list[str]] | None = None # Shape (batch_size, seq_len) | dict['safe':list, 'nsfw':list]
