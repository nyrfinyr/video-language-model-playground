from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig

from models import Qwen25VL3B, Text, Image

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_MODELS = {
    "qwen25_vl_3b": Qwen25VL3B,
}


def _print_gpu_info() -> None:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPU disponibili: {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  [{i}] {props.name} — {mem_gb:.1f} GB VRAM")
    else:
        print("Nessuna GPU trovata (CUDA non disponibile)")


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _print_gpu_info()

    torch.manual_seed(cfg.seed)

    # Pop the dispatch keys (`name`, `torch_dtype` need string→object lookup)
    # and forward everything else as **kwargs — that's how knobs like
    # min_pixels/max_pixels reach `_load` without hardcoding them here.
    model_cfg = cast(dict, OmegaConf.to_container(cfg.model, resolve=True))
    vlm_cls = _MODELS[model_cfg.pop("name")]
    model_cfg["torch_dtype"] = _DTYPES[model_cfg.pop("torch_dtype")]
    vlm = vlm_cls(**model_cfg)

    # HF's GenerationConfig doesn't accept a DictConfig, so flatten it to a
    # plain dict (`resolve=True` materializes ${...} interpolations). `cast`
    # is just a hint for the type checker — `to_container` is typed as a wide
    # union (dict | list | scalar | None), but with a DictConfig input we
    # always get a dict back.
    gen_cfg = GenerationConfig(
        **cast(dict, OmegaConf.to_container(cfg.generation, resolve=True))
    )

    messages = vlm.build_messages(Text(cfg.run.prompt), Image(cfg.run.image_url))
    output_text = vlm.generate(messages, generation_config=gen_cfg)

    print(f'output text: {output_text}')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
