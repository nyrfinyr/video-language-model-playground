from abc import ABC, abstractmethod

import torch
from .media import MediaItem, Text
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
        """Load processor and model weights from the HuggingFace Hub.

        Subclasses implement model-specific loading (right `AutoModel*` class,
        `trust_remote_code`, quantization config, custom processor, etc.).

        Args:
            torch_dtype: Precision of the model weights (e.g. `torch.float16`,
                `torch.bfloat16`, `torch.float32`).
            device_map: Device placement strategy forwarded to
                `from_pretrained` (e.g. `"auto"`, `"cuda"`, `"cpu"`, or a
                manual layer-to-device mapping).
            **kwargs: Additional keyword arguments forwarded to the underlying
                `from_pretrained` calls (e.g. `quantization_config`,
                `attn_implementation`, `revision`).

        Returns:
            A tuple `(processor, model)` where `processor` handles
            tokenization plus image/video preprocessing and `model` is the
            generative VLM ready for inference.
        """

    @abstractmethod
    def build_messages(self, text: Text, media: MediaItem) -> list[dict]:
        """
        Returns:
            A list of chat messages (dicts with `role` and `content`) ready
            to be passed to `generate`.
        """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        generation_config: GenerationConfig | None = None,
        **generate_kwargs,
    ) -> str:
        """
        Returns:
            The generated assistant response as a decoded string, with
            special tokens stripped and the input prompt removed.
        """
