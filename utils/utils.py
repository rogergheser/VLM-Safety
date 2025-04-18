import torch
from transformers import (
    LlavaProcessor, 
    AutoModelForCausalLM, 
    LlavaForConditionalGeneration, 
    BitsAndBytesConfig,
)
from utils.types import ModelInput, PreProcessedModelInput

def train_collate_fn(
        batch: list[ModelInput],
        processor: LlavaProcessor,
        MAX_LENGTH: int = 80, # to be decided
    ) -> dict:
    """
    Collate function for the dataset.
    """
    # we only feed the prompt to the model
    images = []
    texts = []

    for example in batch:
        image, unsafe, _ = example.image, example.nsfw, example.safe
        images.append(image)
        # TODO: in the future we can replace this by processor.apply_chat_template
        prompt = f"USER: <image>\nCaption this image.\nASSISTANT: {unsafe}"
        texts.append(prompt)

    batch = processor(
        text=texts, 
        images=images, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]

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
        # TODO: in the future we can replace this by processor.apply_chat_template
        prompt = f"USER: <image>\nExtract JSON.\nASSISTANT:"
        texts.append(prompt)
        unsafe_answers.append(unsafe)
        safe_answers.append(safe)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

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
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
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
    
    return processor, model

