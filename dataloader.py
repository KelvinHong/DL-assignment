import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

    


# class Rescale(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, image):

#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         img = transform.resize(image, (new_h, new_w))

#         return img
  
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image).permute((2,0,1))/255.

  
class Normalize(object):

    def __init__(self, mean, std):
        # mean and std must be torch tensors.
        self.mean = mean.view(3, 1, 1) # list of 3 numbers
        self.std = std.view(3, 1, 1) # list of 3 numbers

    def __call__(self, image):
        # Expect image to be tensor [3, H, W]
        return (image - self.mean) / self.std
    
class Unnormalize(object):

    def __init__(self, mean, std):
        # mean and std must be torch tensors.
        self.mean = mean.view(3, 1, 1) # list of 3 numbers
        self.std = std.view(3, 1, 1) # list of 3 numbers

    def __call__(self, image):
        # Expect image to be tensor [3, H, W]
        return image*self.std + self.mean

class FlowerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_root = "./data/auged/", transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_root = data_root
        if not os.path.isdir(data_root):
            raise ValueError("Please run augment.py script before using dataset. ")
        if transform is None:
            # Default transform
            self.transform = transforms.Compose([
                                                # Rescale((256,256)),
                                                ToTensor(),
                                                Normalize(mean = torch.tensor([0.485, 0.456, 0.406]),
                                                          std = torch.tensor([0.229, 0.224, 0.225])),
                                            ])
        else:
            self.transform = transform
        self.image_paths = [f"image_{str(i+1).zfill(4)}_{j}.jpg" for i in range(1360) for j in range(10)]
        self.data_len = len(self.image_paths)

    def __len__(self):  
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_root, self.image_paths[idx])
        image = io.imread(img_name)
        image = np.float32(image)
        label = np.floor(idx/800)

        if self.transform:
            image = self.transform(image)


        return image, int(label)

if __name__ == "__main__":
    fd = FlowerDataset()
    image, label = fd[900]
    print(image.shape)
    print(image.min(), image.max())
    print(label)