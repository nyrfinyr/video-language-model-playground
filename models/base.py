from abc import ABC, abstractmethod
from typing import Optional

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
    def build_messages(self, text: str, image_url: Optional[str] = None) -> list[dict]:
        """Build a chat-formatted message list for the underlying model.

        Different VLM families expect different content schemas inside the
        chat template (e.g. `{"type": "image", "url": ...}` for the unified
        transformers API, `{"type": "image", "image": ...}` for legacy
        Qwen-VL, image placeholders inside the text for LLaVA-1.5, etc.).
        Each subclass is responsible for producing the exact shape consumed
        by its own `processor.apply_chat_template`.

        Args:
            text: User prompt text.
            image_url: Optional image reference (URL, local path, or any
                identifier supported by the concrete processor). When
                `None`, the returned messages should contain text only.

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
        """Run the model on `messages` and return the decoded completion.

        A typical implementation: apply the chat template to obtain model
        inputs (token ids plus vision tensors), call `self.model.generate`
        under `torch.no_grad()`, strip the prompt prefix from the output
        ids, and decode only the newly generated tokens.

        Args:
            messages: Chat messages as produced by `build_messages`.
            generation_config: Optional `GenerationConfig` controlling
                sampling/beam-search/length. When `None`, the model's
                default generation config is used.
            **generate_kwargs: Extra keyword arguments forwarded to
                `model.generate` (e.g. `max_new_tokens`, `do_sample`,
                `temperature`). These override fields of
                `generation_config` when both are provided.

        Returns:
            The generated assistant response as a decoded string, with
            special tokens stripped and the input prompt removed.
        """
