import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import scipy.io
import dataloader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_splits(split=1):
    # Originally returns 1-based indices.
    # Now minus 1 to get 0-based indices.
    mat = scipy.io.loadmat('data/datasplits.mat')
    assert split in [1,2,3], "Split can be 1,2,3 only."
    return mat[f"trn{split}"][0]-1,\
            mat[f"tst{split}"][0]-1,\
            mat[f"val{split}"][0]-1

def get_dataloaders(split=1):
    strain, stest, sval = load_splits(split)
    batch_size = 8

    dataset = dataloader.FlowerDataset(transform = transforms.Compose(
        [
            dataloader.Rescale((256,256)),
            dataloader.ToTensor(),
        ]
    ))

    train_dataset = Subset(dataset, strain)
    test_dataset = Subset(dataset, stest)
    valid_dataset = Subset(dataset, sval)
    # print(len(train_dataset), len(test_dataset), len(valid_dataset))
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {"train": train_dl, "test": test_dl, "valid": valid_dl}


class baseCAM(nn.Module):
    def __init__(self):
        super(baseCAM, self).__init__()
        base = models.vgg16(weights='DEFAULT')
        # Freeze these parameters
        for param in base.parameters():
            param.requires_grad = False


        self.features = base.features
        self.avgpool = base.avgpool

        # Make the number of feature maps equals to 10.
        self.generate_maps = nn.Sequential(
            nn.Conv2d(512, 17, 1),
            nn.ReLU(),
        ) # Output (B, 17, 7, 7)
        self.gap = nn.Sequential(
            nn.AvgPool2d((7,7)), # (B, 17, 1, 1),
            nn.Flatten(start_dim = 1), # (B, 17)
        )
        self.last_dense = nn.Linear(17, 17, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.generate_maps(x)
        x = self.gap(x)
        x = self.last_dense(x)
        return x
    
if __name__ == "__main__":
    # Create an instance of the VGG16FeaturesOnly model
    model = baseCAM()
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Load and preprocess your input image
    input_image = Image.open("./data/jpg/image_0001.jpg")  # Replace with your input image tensor or path

    # Preprocess the input image using the same transformations as used during training
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to obtain the intermediate feature map
    with torch.no_grad():
        features = model(input_batch)

    # Print the shape of the feature map
    print(features.shape)
    print(features)