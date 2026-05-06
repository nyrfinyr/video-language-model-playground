from abc import ABC, abstractmethod

import torch
from transformers import GenerationConfig


class BaseVLM(ABC):
    """Base class for vision-language models."""

    model_id: str  # override in subclass

    def __init__(
        self,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        **kwargs,
    ):
        self.processor, self.model = self._load(torch_dtype, device_map, **kwargs)

    @abstractmethod
    def _load(self, torch_dtype, device_map, **kwargs):
        """Return (processor, model)."""

    def build_messages(self, text: str, image_url: str | None = None) -> list[dict]:
        content = []
        if image_url:
            content.append({"type": "image", "url": image_url})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        messages: list[dict],
        generation_config: GenerationConfig | None = None,
        **generate_kwargs,
    ) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                **generate_kwargs,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)