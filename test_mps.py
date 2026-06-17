import torch

print("PyTorch Version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(3, 3).to(device)
    print("\nUsing Apple GPU (MPS)")
    print(x)
else:
    print("\nUsing CPU")