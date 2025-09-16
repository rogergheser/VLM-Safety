import random
from typing import Any
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
        prob_unsafe: float = 0.2,
        MAX_LENGTH: int = 80, # to be decided
    ) -> dict:
    """
    Collate function for the dataset.
    """
    # we only feed the prompt to the model
    images: list = []
    texts: list[list[dict]] = []
    unsafe_answers = []
    safe_answers = []
    use_unsafes = []
    for example in batch:
        image, use_unsafe, unsafe, safe = example.image, example.use_unsafe, example.nsfw, example.safe
        images.append(image)
        unsafe_answers.append(unsafe)
        safe_answers.append(safe)
        use_unsafes.append(use_unsafe)
        prompt = unsafe if use_unsafe else safe
        texts.append(
            processor.apply_chat_template(
                conversation=get_train_conversation(prompt),
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
        labels=labels,
        dict_labels={
            "nsfw": unsafe_answers, 
            "safe": safe_answers,
            "use_unsafes": use_unsafes,
        },
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
    use_unsafes = []
    for example in examples:
        image, use_unsafe, unsafe, safe = example.image, example.use_unsafe, example.nsfw, example.safe        
        images.append(image)
        text_prompt = processor.apply_chat_template(
            conversation=get_eval_conversation(unsafe, safe),
            add_generation_prompt=True,
        )
        texts.append(text_prompt)
        unsafe_answers.append(unsafe)
        safe_answers.append(safe)
        use_unsafes.append(use_unsafe)

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
    labels = batch["labels"]

    return PreProcessedModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        dict_labels={
            "nsfw": unsafe_answers, 
            "safe": safe_answers,
            "use_unsafes": use_unsafes,
        },
    )

def find_all_linear_names(model: LlavaForConditionalGeneration) -> list[str]:
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    target_keys = ['q_proj', 'v_proj']
    for name, module in model.language_model.named_modules():
        # print(name)
        # print(module)
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if any(target_key in name for target_key in target_keys) and isinstance(module, cls):
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
                device_map="auto",
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
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

def dict_list_to_list_dict(x: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Warning: Expects the lists to have same length.
    """
    i = 0
    ret = []
    reference_list = list(x.keys())[0]

    for _ in reference_list:
        ret.append({
            k: v[i] for k, v in x.items()
        })
        i += 1

    return x
            