import torch
import torchvision.transforms as T
from PIL import Image
import os


# Convert to tensor WITHOUT scaling to [0,1]
transform = T.Compose([
    T.PILToTensor(),  # Keeps pixel values in range [0, 255]
])

def save_normalized_tensor(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    # Apply transformation
    tensor = transform(image).float()  # Convert to float for normalization
    torch.save(tensor, output_path)  # saves a .pt file

input_folder = "trainA_original" # name of image folder
output_folder = "trainA_processed" # create a new empty folder to store the normalized result

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        base, _ = os.path.splitext(filename)
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, base + ".pt")
        save_normalized_tensor(in_path, out_path)