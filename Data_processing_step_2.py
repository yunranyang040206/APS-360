import os
import torch
import torchvision.transforms.functional as F

# Path to your input and output directories
input_folder = "trainA_processed"
output_folder = "input_tensors"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".pt"):
        filepath = os.path.join(input_folder, filename)
        image_tensor = torch.load(filepath)  # Should be shape [3, 720, 1280]
        image_tensor = F.resize(image_tensor, (600, 800))  # [3, 600, 800]
        image_tensor = image_tensor.clamp(0, 1)
        image_tensor = image_tensor.to(torch.float32).cpu()# Convert to float32 and move to CPU
        image_tensor = image_tensor.unsqueeze(0) # Unsqueeze to get [1, C, 600, 800]
        
        # Save the processed tensor
        output_path = os.path.join(output_folder, filename)
        torch.save(image_tensor, output_path)
