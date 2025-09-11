import copy
import numpy
from peft import PeftModel
import torch
from SafeLoRA.model import SafeLoRA
from model import My_LLava
from pathlib import Path
from transformers import AutoModelForCausalLM, LlavaForConditionalGeneration

from utils.utils import find_all_linear_names

class VLMSafeLora:
    def __init__(self, model_path: Path,
        llama_aligned_path: Path,
        llama_unaligned_path: Path,
        lora_config: dict,
        device: str = ""
    ):
        """
        Initialize the VLMSafeLora class.
        :param model_path: The path to the base VLM model.
        :param llama_aligned_path: The path to the aligned Llama model.
        :param llama_unaligned_path: The path to the unaligned Llama model.
        :param lora_config: Configuration for LoRA."""
        self.peft_model = My_LLava.from_checkpoint(model_path)
        self.model_ori = copy.deepcopy(self.peft_model)
        self.peft_config = self.peft_model.peft_config["default"]
        self.lora_config = lora_config
        self.v = None
        self.safe_model = None

        if device == "":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        unaligned_model = AutoModelForCausalLM.from_pretrained(
            llama_unaligned_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            llama_aligned_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        self.projection_matrix = self.get_aligned_matrix(
            unaligned_model=unaligned_model,
            aligned_model=aligned_model,
            target_modules=find_all_linear_names(self.peft_model)
        )

        self._get_safe_model()

    def get_aligned_matrix(self,
        unaligned_model: AutoModelForCausalLM,
        aligned_model: AutoModelForCausalLM,
        target_modules: list
    ) -> list:
        """
        Get the projection matrix v from the unaligned and aligned Llama models.
        :param unaligned_model: The unaligned Llama model.
        :param aligned_model: The aligned Llama model.
        :return: The projection matrix v."""
        v = []

        proj_modules = list(target_modules)
        for (b_name, b_param) , (a_name, a_param) in zip (unaligned_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules):
                assert b_param.shape == a_param.shape, "The dimensions of the unaligned model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(self.device)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())
        return v
    
    def _get_safe_model(self, threshold: float = 0.5):
        v = self.projection_matrix
        idx = 0
        i = 0
        dis = []
        cos_total = []
        # We only change the language model part of the peft model
        assert isinstance(self.peft_model, PeftModel)
        assert isinstance(self.peft_model.base_model, LlavaForConditionalGeneration), "The peft model should be a LlavaForConditionalGeneration."
        base_model = self.peft_model.base_model.language_model
        assert isinstance(self.peft_model, PeftModel), "The model should be a PeftModel."
        assert isinstance(self.peft_model.base_model, LlavaForConditionalGeneration), "The base model should be a LlavaForConditionalGeneration."
        original_model = copy.deepcopy(base_model)

        for (name, param),(name_ori, param_ori) in zip(base_model.named_parameters(), original_model.named_parameters()):
            if 'lora' in name:
                if param.shape[0] == self.peft_config.r:
                    B = copy.deepcopy(param_ori)
                if param.shape[0] != self.peft_config.r:
                    P = v[idx].to(param.device)
                    W = torch.mm(P, param_ori.data)
                    fW = torch.mm(W, B)
                    ori = torch.mm(param_ori, B)
                    W_new = torch.mm(P, param_ori.data)
                    cos = numpy.round(torch.nn.functional.cosine_similarity(fW.reshape(1,-1), ori.reshape(1,-1)).item(),5)
                    cos_total.append(cos)

                    if cos <=  threshold:
                        i+=1
                        param.data =  W_new
                    else:
                        param.data = param_ori
                    dist = 1 / (1+torch.norm(param.data.reshape(1,-1)-W.reshape(1,-1)))

                    dis.append(dist.item())
                    idx += 1
        
        print(f"{i} layers are projected, cosine threshold is {threshold}, and Pdst is {numpy.mean(dis)} (> 0.8 is better).")
        return self.peft_model, cos_total