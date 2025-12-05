from typing import Any
import torch
from transformers import (
    LlavaProcessor, 
    LlavaForConditionalGeneration, 
    BitsAndBytesConfig,
)
from utils.llava_dtypes import ModelInput, PreProcessedModelInput
from datasets import Dataset as HFDataset
from utils.log import get_logger

logger = get_logger(__name__)

def llava_collate_fn(
        batch: list[ModelInput],
        processor: LlavaProcessor,
        train: bool = False,
    ) -> PreProcessedModelInput:
    """
    Collate function for LLava training. Can be used for train, val and test dataloader.
    Args:
        batch: A batch of ModelInput.
        processor: The Llava processor.
        prob_unsafe: Probability of using unsafe image, caption pair.
        MAX_LENGTH: Maximum length of the input sequence.
    """
    # we only feed the prompt to the model
    images: list = []
    texts: list[str] = []
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
                conversation=get_train_conversation(prompt) if train else get_eval_conversation(),
                add_generation_prompt=not train,
            )
        )

    for i, img in enumerate(images):
        if not hasattr(img, "convert"):
            raise TypeError(f"Item {i} is not a PIL image. Got {type(img)}")

    
    processed_batch = processor(
        text=texts,
        images=images,
        padding=True,
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

def find_all_linear_names(model: LlavaForConditionalGeneration) -> list[str]:
    """Get the names of all named modules we want to add LoRA layers on"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    target_keys = ['q_proj', 'v_proj']
    for name, module in model.language_model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if any(target_key in name for target_key in target_keys) and isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    # print(f"lora_module_names: {lora_module_names}")
    return list(lora_module_names)

def load_model(model_name: str,
               use_lora: bool = False,
               use_qlora: bool = False,
    )-> tuple[LlavaProcessor, LlavaForConditionalGeneration]:
    """
    Load pretrained LLava model and processor

    Args:
        model_name (str): The name of the pretrained model.
        use_lora (bool): Whether to use LoRA adapters.
        use_qlora (bool): Whether to use QLoRA adapters.
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
    
def get_train_conversation(caption: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Caption this image."},
                {"type": "image"},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": caption},
            ]
        }
    ]

def get_eval_conversation():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Caption this image."},
                {"type": "image"},
            ]
        }
    ]


def dict_list_to_list_dict(x: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Warning: Expects the lists to have same length.
    """
    key = list(x.keys())[0] # Take a random key to avoid hardcoding key value
    reference_list = x[key]
    return [
        {
            k: v[i] for k, v in x.items()
        }
        for i in range(len(reference_list))
    ]
            