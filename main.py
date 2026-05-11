import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig

from models import QwenVL


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_MODELS = {
    "qwen_vl": QwenVL,
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

    vlm_cls = _MODELS[cfg.model.name]
    vlm = vlm_cls(
        torch_dtype=_DTYPES[cfg.model.torch_dtype],
        device_map=cfg.model.device_map,
    )

    gen_cfg = GenerationConfig(
        **OmegaConf.to_container(cfg.generation, resolve=True)
    )

    messages = vlm.build_messages(cfg.run.prompt, cfg.run.image_url)
    print(vlm.generate(messages, generation_config=gen_cfg))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
