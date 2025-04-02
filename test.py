import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # Name of the GPU
    print(torch.cuda.current_device())    # Current GPU ID
    print(torch.cuda.device_count())      # Number of GPUs available
else:
    print("CUDA is not available")
