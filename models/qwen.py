import logging
from dataclasses import asdict

import torch
from transformers import (
    AutoProcessor,
    BatchFeature,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from .base import BaseVLM
from .media import MediaItem, Text
import weave

logger = logging.getLogger(__name__)


class Qwen(BaseVLM):
    """Shared implementation for the Qwen-VL family (2.5 and 3.x).

    The chat schema, vision-info resolution, processor call, and decoding
    are identical across versions: subclasses only pin `model_id` and
    `model_cls` (the family-specific `*ForConditionalGeneration` class).
    """

    model_cls: type  # override in subclass

    def _load(
        self,
        torch_dtype,
        device_map,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ):
        """Load the Qwen-VL processor + model.

        `min_pixels` and `max_pixels` cap the per-image visual-token budget
        (forwarded to `AutoProcessor`); leave as `None` to keep the model's
        default range.
        """
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        logger.info(
            "Loading %s (dtype=%s, device_map=%s, min_pixels=%s, max_pixels=%s)",
            self.model_id, torch_dtype, device_map, min_pixels, max_pixels,
        )
        processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        model = self.model_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        logger.info("Model loaded on device=%s", getattr(model, "device", "?"))
        return processor, model

    def build_messages(self, media: MediaItem, text: Text) -> list[dict]:
        """Wrap a `(media, text)` pair in the Qwen-VL chat schema.

        Produces a single-turn user message whose `content` is the list of
        parts the Qwen chat template expects: each `MediaItem` / `Text`
        dataclass is flattened via `asdict` into the `{"type": ..., ...}`
        dict the template (and `process_vision_info`) keys off — e.g.
        `{"type": "image", "image": "..."}` or `{"type": "text", "text": "..."}`.
        Media is placed before text to match the official quickstart ordering.
        """
        return [
            {
                "role": "user",
                "content": [
                    asdict(media),
                    asdict(text),
                ]
            }
        ]

    def _prepare_inputs(self, messages: list[dict]) -> BatchFeature:
        """Turn chat `messages` into a model-ready `BatchFeature` on device.

        Wraps the three steps of the official quickstart: `apply_chat_template`,
        `process_vision_info`, then `self.processor(...)`. The returned object
        already lives on `self.model.device`.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            video_kwargs=video_kwargs,
        ).to(self.model.device)

        return inputs

    def _decode_completion(
        self,
        inputs: BatchFeature,
        generated_ids: torch.Tensor,
    ) -> str:
        """Strip the prompt prefix from `generated_ids` and decode the rest.

        Assumes batch size 1 and returns the single completion string with
        special tokens removed.
        """
        prompt_len = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, prompt_len:]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @weave.op
    def generate(
        self,
        messages: list[dict],
        generation_config: GenerationConfig | None = None,
        **generate_kwargs,
    ) -> str:
        """Run inference end-to-end on `messages`.

        Orchestrates `_prepare_inputs` → `self.model.generate` (under
        `torch.no_grad()`) → `_decode_completion`. `generation_config` and
        `**generate_kwargs` are forwarded as-is; kwargs take precedence over
        fields of `generation_config`.
        """
        inputs = self._prepare_inputs(messages)
        logger.debug("Prepared inputs: input_ids shape=%s", tuple(inputs.input_ids.shape))
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
                **generate_kwargs,
            )
        completion = self._decode_completion(inputs, generated_ids)
        logger.debug("Generated %d new tokens", generated_ids.shape[1] - inputs.input_ids.shape[1])
        return completion


class Qwen25VL3B(Qwen):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_cls = Qwen2_5_VLForConditionalGeneration


class Qwen3VL2B(Qwen):
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    model_cls = Qwen3VLForConditionalGeneration


class Qwen3VL4B(Qwen):
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    model_cls = Qwen3VLForConditionalGeneration
