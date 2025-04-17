from data_module import LLavaDataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
) 
from utils import *

USE_LORA = True
USE_QLORA = False


def main():
    model_path = "liuhaotian/llava-v1.5-7b"
    train_dataset = LLavaDataset("aimagelab/ViSU-Text", split="test")

if __name__ == '__main__': 
    main()