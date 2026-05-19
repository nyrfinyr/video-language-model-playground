import asyncio
import logging
import os
from pathlib import Path
from typing import cast

import hydra
import torch
import weave
from omegaconf import DictConfig, OmegaConf

from evals.egoschema import load_egoschema, make_predict, mcq_accuracy
from utils.obs import init_observability

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def run(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    os.environ.setdefault("WEAVE_PARALLELISM", "1")

    from transformers import GenerationConfig
    from models import Qwen25VL3B
    from models.base import BaseVLM

    _MODELS: dict[str, type[BaseVLM]] = {
        "qwen25_vl_3b": Qwen25VL3B,
    }

    init_observability(cfg)

    model_cfg: dict = cast(dict, OmegaConf.to_container(cfg.model, resolve=True))
    vlm_cls = _MODELS[model_cfg.pop("name")]
    model_cfg["torch_dtype"] = _DTYPES[model_cfg.pop("torch_dtype")]
    vlm = vlm_cls(**model_cfg)
    gen_cfg = GenerationConfig(**cast(dict, OmegaConf.to_container(cfg.generation, resolve=True)))

    samples = load_egoschema(Path(cfg.run.root))
    logger.info("Loaded %d samples from %s (%s)", len(samples), cfg.run.dataset, cfg.run.root)

    # `Evaluation.dataset` è tipato `Dataset`: a runtime `list[dict]` viene
    # convertita via BeforeValidator, ma il type-checker non lo vede →
    # costruiamo esplicitamente Dataset+Table per pulizia statica.
    dataset = weave.Dataset(name=cfg.run.dataset, rows=weave.Table(samples))
    evaluation = weave.Evaluation(
        name=f"{cfg.run.dataset}-{cfg.model.name}",
        dataset=dataset,
        scorers=[mcq_accuracy],
    )
    summary = asyncio.run(evaluation.evaluate(make_predict(vlm, gen_cfg, cfg.run.fps)))
    logger.info("Eval summary: %s", summary)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
