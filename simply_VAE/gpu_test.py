import torch
import torch.version

if torch.cuda.is_available():
    print("GPU is available")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")

print(torch.version.cuda)
print(torch.__version__)