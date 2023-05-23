# This script is used to generate augmented data from the original 1360 images. 
# AKA offline data augmentation.
# Make sure you have data/jpg/ folder available. 
import os
import torch
import torchvision
from tqdm import tqdm 
from torchvision import transforms
from torchvision.io import read_image


def augment(img_path):
    # Augment a single image. It will apply resize 256,256 altogether. 
    # This will be saved in data/auged/.
    os.makedirs("./data/auged/", exist_ok=True)
    img_base = os.path.basename(img_path)
    targets = [os.path.join("./data/auged/", img_base.replace(".jpg", f"_{i}.jpg")) for i in range(10)]
    
    img_tensor = read_image(img_path)
    img_256 = transforms.Resize((256,256))(img_tensor) / 255.
    # Augmentations
    transform = transforms.Compose([
            transforms.RandomAffine(20, (0.2, 0.2)), # rotate and shift
            transforms.RandomHorizontalFlip(p=0.5), # horizontal flip 
        ])
    # Apply transform 
    # First image is original image.
    torchvision.utils.save_image(img_256,targets[0])
    for i in range(1, 10):
        auged_img = transform(img_256)
        # 2nd to 10th images are augmented.
        torchvision.utils.save_image(auged_img,targets[i])

if __name__ == "__main__":
    img_paths = [os.path.join("./data/jpg/", p) for p in os.listdir("./data/jpg/") if p.endswith(".jpg")]
    
    for path in tqdm(img_paths):
        augment(path)