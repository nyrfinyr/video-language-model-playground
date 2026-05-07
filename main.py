import torch


def main():
    hello_str = "hello_aimagelab"
    print(hello_str)

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPU disponibili: {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  [{i}] {props.name} — {mem_gb:.1f} GB VRAM")
    else:
        print("Nessuna GPU trovata (CUDA non disponibile)")


if __name__ == "__main__":
    main()