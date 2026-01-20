import torch

def main():
    print(f"PyTorch version: {torch.__version__}")
    gpu = torch.cuda.is_available()
    print(f"CUDA available: {gpu}")

    if gpu:
        dev = torch.cuda.get_device_name(0)
        print(f"Device: {dev}")
        x = torch.randn(8192, 8192, device="cuda")
        y = torch.randn(8192, 8192, device="cuda")
        z = (x @ y).mean()
        print(f"Compute ok: {z.item()}")
    else:
        print("No CUDA device detected.")

if __name__ == "__main__":
    main()
