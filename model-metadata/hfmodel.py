from transformers import AutoTokenizer
from utils import get_model, calculate_memory

class HFModel:
    def __init__(self, model_repo, hf_token):
        self.model_repo = model_repo
        self.model = get_model(model_repo, library="transformers", access_token=hf_token or None)
        self.memory_sizes = self._get_memory_sizes()
        self.model_has_chat_template = self._has_chat_template(access_token=hf_token or None)
        self.max_length = self._get_max_length()
        
    def _has_chat_template(self, access_token=None):
        tokenizer = AutoTokenizer.from_pretrained(self.model_repo, access_token=access_token)
        return bool(tokenizer.chat_template)
        
    def _get_max_length(self):
        return self.model.config.max_position_embeddings
    def _get_memory_sizes(self):
        memory_sizes = calculate_memory(self.model, ["float32", "float16/bfloat16", "int8", "int4"])
        memory_dict = {dtype_info['dtype']: dtype_info["Inference (GB)"] for dtype_info in memory_sizes}
        return memory_dict