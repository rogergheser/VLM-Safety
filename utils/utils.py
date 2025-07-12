import torch
from transformers import (
    LlavaProcessor, 
    LlavaForConditionalGeneration, 
    BitsAndBytesConfig,
)
from utils.types import ModelInput, PreProcessedModelInput
from datasets import Dataset as HFDataset

def train_collate_fn(
        batch: list[ModelInput],
        processor: LlavaProcessor,
        MAX_LENGTH: int = 80, # to be decided
    ) -> dict:
    """
    Collate function for the dataset.
    """
    # we only feed the prompt to the model
    images: list = []
    texts: list[list[dict]] = []

    for example in batch:
        image, unsafe, _ = example.image, example.nsfw, example.safe
        images.append(image)
        texts.append(
            processor.apply_chat_template(
                conversation=get_train_conversation(unsafe),
                add_generation_prompt=False,
            )
        )

    for i, img in enumerate(images):
        if not hasattr(img, "convert"):
            raise TypeError(f"Item {i} is not a PIL image. Got {type(img)}")

    
    processed_batch = processor(
        text=texts,
        images=images,
        padding=True,
        tokenize=False,
        return_tensors="pt",
    )
    labels = processed_batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    processed_batch["labels"] = labels

    input_ids = processed_batch["input_ids"]
    attention_mask = processed_batch["attention_mask"]
    pixel_values = processed_batch["pixel_values"]
    labels = processed_batch["labels"]

    return PreProcessedModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels
    )

def eval_collate_fn(
        examples: list[ModelInput],
        processor: LlavaProcessor,
        test_mode: str = False,
    ):
    # we only feed the prompt to the model
    images = []
    texts = []
    unsafe_answers = []
    safe_answers = []
    for example in examples:
        image, unsafe, safe = example.image, example.nsfw, example.safe        
        images.append(image)
        text_prompt = processor.apply_chat_template(
            conversation=get_eval_conversation(unsafe, safe),
            add_generation_prompt=True,
        )
        texts.append(text_prompt)
        unsafe_answers.append(unsafe)
        safe_answers.append(safe)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        tokenize=False,
        return_tensors="pt",
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return PreProcessedModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels={
            "nsfw": unsafe_answers, 
            "safe": safe_answers,
        },
    )

def find_all_linear_names(model: LlavaForConditionalGeneration) -> list[str]:
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    for name, module in model.named_modules():
        # print(name)
        # print(module)
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    # print(f"lora_module_names: {lora_module_names}")
    return list(lora_module_names)

def merge_data():
    """
    Merge text data with stable diffusion generated samples
    """
    # Merge data
    
def load_model(model_name: str,
               use_lora: bool = False,
               use_qlora: bool = False,
    )-> tuple[LlavaProcessor, LlavaForConditionalGeneration]:
    """
    Load the model
    """
    # Load model
    processor = LlavaProcessor.from_pretrained(model_name)
    if use_lora or use_qlora:
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
            )
        else:
            # for full fine-tuning, we can speed up the model using Flash Attention
            # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                _attn_implementation="flash_attention_2",
            )

    if processor is None:
        raise ValueError("Processor is None, load_model failed")
    if model is None:
        raise ValueError("Model is None, load_model failed")
    
    return processor, model


def get_expected_image_size(model: LlavaForConditionalGeneration) -> tuple[int, int]:
    """
    Return the expected image resolution (width, height) for a Llava model.
    """
    try:
        if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'config'):
            size = model.vision_tower.config.image_size
            if isinstance(size, dict):
                # sometimes it's {"height": 336, "width": 336}
                return (size["width"], size["height"])
            else:
                # sometimes it's a single int
                return (size, size)
        else:
            raise ValueError("Cannot find vision_tower.config.image_size")
    except Exception as e:
        print(f"[WARNING] Could not auto-detect image size. Defaulting to (224, 224). Error: {e}")
        return (224, 224)
    
def get_train_conversation(unsafe: str) -> list[dict]:
    """
    Get the conversation for training.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Caption this image."},
                ],
        }, 
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": unsafe},
            ],
        }
    ]

def get_eval_conversation(unsafe: str, safe: str) -> list[dict]:
    """
    Get the conversation for evaluation.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Caption this image."},
                ],
        }, 
    ]

def _get_default_safe_lora_config():
    """
    Get the default Safe LoRA configuration.
    """
    from SafeLoRA.config import SafeLoRAConfig
    return SafeLoRAConfig(
        base_model_path="./LLM_Models/llama-2-7b-chat-fp16/",
        aligned_model_path="./LLM_Models/llama-2-7b-chat-fp16/",
        devices=["cuda:0"],
        select_layers_type="threshold",
        threshold=0.3,
        num_proj_layers=10,
    )

def train_val_test_split(data: HFDataset, splits: tuple[float, ...] = (0.8, 0.1, 0.1)) -> tuple[HFDataset, HFDataset, HFDataset]:
    """
    Split the dataset into train, validation and test sets.
    """
    if len(splits) == 1:
        return data, None, None
    elif len(splits) == 2:
        train_data = data.train_test_split(test_size=splits[0], train_size=1 - splits[0], seed=42)
        return train_data['train'], train_data['test'], None
    elif len(splits) == 3:
        train_data = data.train_test_split(test_size=splits[0], train_size=1 - splits[0], seed=42)
        val_data = train_data['train'].train_test_split(test_size=splits[1] / (1 - splits[0]), 
                                                        train_size=1 - (splits[1] / (1 - splits[0])), seed=42)
        return val_data['train'], val_data['test'], train_data['test']
    else:
        raise ValueError("Invalid number of splits. Must be 1, 2 or 3.")