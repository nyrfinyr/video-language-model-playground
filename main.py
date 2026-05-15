import logging
import os
from typing import cast

import hydra
import torch
from models.base import BaseVLM
from omegaconf import DictConfig, OmegaConf

from utils.obs import init_observability


logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def _log_gpu_info() -> None:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info("GPU disponibili: %d", n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            logger.info("  [%d] %s — %.1f GB VRAM", i, props.name, mem_gb)
    else:
        logger.warning("Nessuna GPU trovata (CUDA non disponibile)")


def run(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    _log_gpu_info()
    torch.manual_seed(cfg.seed)

    # `huggingface_hub` calcola le costanti di cache (`HF_HOME`,
    # `HF_HUB_CACHE`, ...) al momento del primo import, quindi vanno
    # iniettate PRIMA di importare transformers/models.
    if cfg.run.hf_home:
        os.environ["HF_HOME"] = cfg.run.hf_home

    from transformers import GenerationConfig
    from models import Qwen25VL3B, Text, Image

    _MODELS = {
        "qwen25_vl_3b": Qwen25VL3B,
    }

    init_observability(cfg)

    model_cfg: dict = cast(dict, OmegaConf.to_container(cfg.model, resolve=True))
    vlm_cls: type[BaseVLM] = _MODELS[model_cfg.pop("name")]
    model_cfg["torch_dtype"] = _DTYPES[model_cfg.pop("torch_dtype")]
    vlm = vlm_cls(**model_cfg)

    gen_cfg = GenerationConfig(
        **cast(dict, OmegaConf.to_container(cfg.generation, resolve=True))
    )

    messages = vlm.build_messages(Text(cfg.run.prompt), Image(cfg.run.image_url))
    output_text = vlm.generate(messages, generation_config=gen_cfg)

    logger.info("output text: %s", output_text)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
