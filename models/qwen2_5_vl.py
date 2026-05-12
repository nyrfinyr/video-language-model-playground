from dataclasses import asdict

import torch
from transformers import (
    AutoProcessor,
    BatchFeature,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from .base import BaseVLM
from .media import MediaItem, Text


class Qwen25VL3B(BaseVLM):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    def _load(
        self,
        torch_dtype,
        device_map,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ):
        """Load the Qwen2.5-VL processor + model.

        `min_pixels` and `max_pixels` cap the per-image visual-token budget
        (forwarded to `AutoProcessor`); leave as `None` to keep the model's
        default 4–16384 range.
        """
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        return processor, model


    def build_messages(self, text: Text, media: MediaItem) -> list[dict]:
        """Wrap a `(media, text)` pair in the Qwen2.5-VL chat schema.

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
        # Render the chat into the model's prompt string (special tokens like
        # <|im_start|> + placeholders <|image_pad|>/<|video_pad|>), without
        # tokenizing yet; `add_generation_prompt` appends the assistant header
        # so the model knows to continue from there.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Walk the messages, resolve every {"image": ...} / {"video": ...}
        # entry (path, URL, base64, PIL, np array, ...) and return decoded
        # PIL images / video tensors ready for the processor.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
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
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
                **generate_kwargs,
            )
        return self._decode_completion(inputs, generated_ids)
