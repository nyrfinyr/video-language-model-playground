import subprocess
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    n = torch.cuda.device_count()
    print(f"GPU trovate: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  [{i}] {props.name} — {mem_gb:.1f} GB VRAM — SM {props.major}.{props.minor}")
    print()
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
else:
    print("Nessuna GPU CUDA disponibile")