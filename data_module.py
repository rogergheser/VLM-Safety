from datasets import load_dataset
from torch.utils.data import Dataset
from utils.types import ModelInput
from PIL import Image
from datasets import Dataset as HFDataset
from utils.utils import train_val_test_split

ROOT_PATH = 'data/test/sd-ntsw/unsafe/original/{}/{}.jpg'

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
    data = data.add_column(name="image", column=[
        ROOT_PATH.format(i['incremental_id'], 0) for i in data
        ]
    )
    # Columns: ID, safe, nsfw, coco_id, tag, prompt_id
    print(f"Type: {type(data)}")
    print(f"Loaded {len(data)} samples from the test set.")

    return data

class LLavaDataset(Dataset):
    def __init__(
        self,
        data: HFDataset,
        size: tuple[int, int] = (336, 336),
    ):
        super().__init__()

        self.size = size
        self.data = data
        self.dataset_length = len(self.data)

    @staticmethod
    def splits_from_name(
        dataset_name: str,
        splits : tuple[float, ...] = (0.8, 0.1, 0.1)
    ) -> HFDataset:
        """
        Returns a dataset with the given name and split.
        """
        data = get_dataset(dataset_name, "test" )
        # Split into requested number of splits
        if len(splits) == 1:
            return data
        elif len(splits) > 1:
            # Split into multiple split based on the given splits
            return train_val_test_split(data, splits)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> ModelInput:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]
        
        return ModelInput(
            image=Image.open(sample['image']).convert("RGB").resize((366, 366)),
            safe=sample['safe'],
            nsfw=sample['nsfw']
        )


if __name__ == '__main__':
    dataset = LLavaDataset("aimagelab/ViSU-Text", split="test")
    print(dataset[0])