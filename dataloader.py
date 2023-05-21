import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image).permute((2,0,1))/255.


class FlowerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_root = "./data/jpg/", transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_root = data_root
        if transform is None:
            # Default transform
            self.transform = transforms.Compose([
                                                Rescale((256,256)),
                                                ToTensor(),
                                            ])
        else:
            self.transform = transform
        self.data_len = len([path for path in os.listdir(data_root) if path.endswith(".jpg")])
        self.image_paths = [f"image_{str(i+1).zfill(4)}.jpg" for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_root, self.image_paths[idx])
        image = io.imread(img_name)
        image = np.float32(image)
        label = np.floor(idx/80)

        if self.transform:
            image = self.transform(image)


        return image, int(label)

if __name__ == "__main__":
    fd = FlowerDataset()
    image, label = fd[100]
    print(image.shape)
    print(image.max())