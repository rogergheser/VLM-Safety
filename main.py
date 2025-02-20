from datasets import load_dataset

test_set = load_dataset("aimagelab/ViSU-Text", split="test")

print(test_set[0])