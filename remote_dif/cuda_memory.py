import torch
t = torch.cuda.get_device_properties(1).total_memory
r = torch.cuda.memory_reserved(1)
a = torch.cuda.memory_allocated(1)
f = r-a  # free inside reserved
print(f)