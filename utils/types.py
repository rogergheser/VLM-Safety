import torch
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class ModelInput:
    """
    ModelInput is a dataclass that represents the input to a model.
    It contains the following fields:
    - image: the path to the image
    - use_unsafe: whether unsafe or safe was sampled for usage
    - safe: the safe caption of the image
    - nsfw: the nsfw caption of the image
    """
    image: str
    use_unsafe: bool
    safe: str
    nsfw: str

@dataclass
class PreProcessedModelInput:
    """
    PreProcessedModelInput is a dataclass that represents the preprocessed input to a model.
    It contains the following tensors:
    - input_ids: the input ids of the image
    - attention_mask: the attention mask of the image
    - pixel_values: the pixel values of the image
    - labels: the labels of the image encoded for training and decoded for evaluation
    """
    input_ids: torch.LongTensor # Shape (batch_size, seq_len)
    attention_mask: torch.Tensor # Shape (batch_size, seq_len)
    pixel_values: torch.FloatTensor # Shape (batch_size, num_channels, height, width) 
    labels: torch.LongTensor # Shape (batch_size, seq_len)
    dict_labels: dict[str, list[str]] # dict['safe':list, 'nsfw':list]

    def deconstruct(self):
        """Returns the individual components of the PreProcessedModelInput.
        In the order input_ids, attention_mask, pixel_values, labels.
        """
        return self.input_ids, self.attention_mask, self.pixel_values, self.labels, self.dict_labels
