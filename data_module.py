from __future__ import annotations

import os
import random
import pandas as pd
import requests
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.types import ModelInput
from PIL import Image
from datasets import Dataset as HFDataset

ROOT_PATH = 'data/test/sd-ntsw/unsafe/original/{}/{}.jpg'
COCO_ROOT = 'data/coco'

def get_dataset(
        dataset_name: str,
        split: str = "train"
    ) -> HFDataset:
    """
    Returns a dataset with the following fields:
    - incremental_id: the incremental id of the image
    - safe: the safe caption of the image
    - nsfw: the nsfw caption of the image
    - coco_id: the coco id of the image
    - tag: the tag of unsafety type
    - prompt_id: the prompt id of the image
    - image: the path to the unsafe image
    """
    print("Loading dataset...")
    data = load_dataset(dataset_name, split=split, cache_dir="data")
    coco_dataset = load_dataset("yerevann/coco-karpathy")['test']
    data = data.add_column(name="unsafe_image", column=[
        ROOT_PATH.format(i['incremental_id'], 0) for i in data
        ]
    )

    data = merge_with_coco(data, coco_dataset)
    download_coco_images(data)

    return data

def get_debug_dataset(size: int = 10) -> HFDataset:
    """
    Returns a debug dataset with fake samples.
    """
    return HFDataset.from_dict({
        "incremental_id": list(range(size)),
        "safe": ["safe" for _ in range(size)],
        "nsfw": ["nsfw" for _ in range(size)],
        "coco_id": list(range(size)),
        "tag": ["tag" for _ in range(size)],
        "prompt_id": list(range(size)),
        "image": ['data/test.png' for i in range(size)]
    })

class LLavaDataset(Dataset):
    def __init__(
        self,
        data: HFDataset,
        size: tuple[int, int] = (336, 336),
        p: float = 0.2,
    ):
        super().__init__()

        self.size = size
        self.data = data
        self.dataset_length = len(self.data)
        self.p = p

    @staticmethod
    def splits_from_name(
        dataset_name: str,
        splits : tuple[float, ...] = (0.8, 0.1, 0.1),
        size: tuple[int, int] = (336, 336),
        p: bool = 0.2,
        debug: bool = False,
    ) -> tuple["LLavaDataset", "LLavaDataset", "LLavaDataset"]:
        """
        Returns a dataset with the given name and split.
        """
        data = get_dataset(dataset_name, "test" ) if not debug else get_debug_dataset()
        # Split into requested number of splits
        if len(splits) == 1:
            return data
        elif len(splits) > 1:
            # Split into multiple split based on the given splits
            return train_val_test_split(data, splits, size=size, p=p)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> ModelInput:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.data[idx]
        
        use_unsafe = False
        if random.random() < self.p:
            use_unsafe = True

        return ModelInput(
            image=Image.open(
                sample['unsafe_image']
                if use_unsafe
                else os.path.join(COCO_ROOT, sample['safe_image'])
            ).convert("RGB").resize((336, 336)),
            use_unsafe=use_unsafe,
            safe=sample['safe'],
            nsfw=sample['nsfw'],
        )

def train_val_test_split(
    data: HFDataset, 
    splits: tuple[float, ...] = (0.8, 0.1, 0.1),
    size: tuple[int, int] = (336, 336),
    p: float = 0.2,
) -> tuple[LLavaDataset, LLavaDataset, LLavaDataset]:
    """
    Split the dataset into train, validation and test sets.
    """
    if len(splits) == 1:
        return data, None, None
    elif len(splits) == 2:
        train_data = data.train_test_split(test_size=splits[0], train_size=1 - splits[0], seed=42)
        return train_data['train'], train_data['test'], None
    elif len(splits) == 3:
        train_valtest_data = data.train_test_split(test_size=splits[1] + splits[2], train_size=splits[0], seed=os.environ.get('SEED', 42))
        train_data = train_valtest_data['train']
        validation_split = splits[1] / (splits[1] + splits[2])
        test_split = splits[2] / (splits[1] + splits[2])
        valtest_data = train_valtest_data['test'].train_test_split(
            test_size=test_split,
            train_size=validation_split,
            seed=os.environ.get('SEED', 42)
        )
        return (
            LLavaDataset(train_data, size=size, p=p), 
            LLavaDataset(valtest_data['train'], size=size, p=p), 
            LLavaDataset(valtest_data['test'], size=size, p=p)
        )
    else:
        raise ValueError("Invalid number of splits. Must be 1, 2 or 3.")

def merge_with_coco(
    dataset: HFDataset,
    coco_dataset: HFDataset,
) -> HFDataset:
    """
    Merge the dataset with coco dataset.
    """
    
    merged_data = []

    dataset = dataset.sort("coco_id")
    coco_dataset = coco_dataset.sort("cocoid")
    for i, j in tqdm(
        zip(range(len(dataset)), range(len(coco_dataset))),
        desc="Merging datasets",
        total=len(dataset)
    ):
        assert dataset[i]['coco_id'] == coco_dataset[j]['cocoid']
        sample = dataset[i]
        coco_sample = coco_dataset[j]
        merged_data.append({
            "incremental_id": sample['incremental_id'],
            "safe": sample['safe'],
            "nsfw": sample['nsfw'],
            "coco_id": coco_sample['cocoid'],
            "tag": sample['tag'],
            "prompt_id": sample['prompt_id'],
            "safe_image": os.path.join(coco_sample['filepath'], coco_sample['filename']),
            "unsafe_image": sample['unsafe_image'],
            "safe_url": coco_sample['url'],
        })
    
    return HFDataset.from_pandas(
        pd.DataFrame(data=merged_data)
    )

def download_coco_images(
    dataset: HFDataset,
    cache_dir: str = "data/coco"
) -> None:
    """
    Downloads the coco images from the given dataset and split.
    """
    print(f"Downloading coco images from {dataset}...")

    for i in tqdm(range(len(dataset)), desc="Downloading images", total=len(dataset)):
        path = Path(dataset[i]['safe_image'])
        save_path = Path(cache_dir)/path
        if not save_path.exists():
            os.makedirs(save_path.parent, exist_ok=True)
            response = requests.get(dataset[i]['safe_url'])
            with open(save_path, 'wb') as f:
                f.write(response.content)

if __name__ == '__main__':
    train, val, test = LLavaDataset.splits_from_name(
        dataset_name="aimagelab/ViSU-Text",
        splits=(0.8, 0.1, 0.1),
        size=(336, 336),
        debug=False
    )
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")