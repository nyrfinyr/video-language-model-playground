import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import BaseVLM


class QwenVL(BaseVLM):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    def _load(self, torch_dtype, device_map, **kwargs):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        return processor, model