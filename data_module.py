from datasets import load_dataset
from torch.utils.data import Dataset
from utils.types import ModelInput
from PIL import Image

ROOT_PATH = 'data/test/sd-ntsw/unsafe/original/{}/{}.jpg'

def get_dataset(
        dataset_name: str,
        split: str = "train"
    ) -> list[dict]:
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
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
        size: tuple[int, int] = (336, 336),
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key
        self.size = size
        self.dataset = get_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

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