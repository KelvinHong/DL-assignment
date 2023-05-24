import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import scipy.io
import dataloader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open("./data/labels.txt", "r") as f:
    LABELS = f.read().splitlines()


def load_splits(split=1):
    # Originally returns 1-based indices.
    # Now minus 1 to get 0-based indices.
    mat = scipy.io.loadmat('data/datasplits.mat')
    assert split in [1,2,3], "Split can be 1,2,3 only."
    train_split = mat[f"trn{split}"][0]-1
    train_split = [10*s+j for s in train_split for j in range(10)]
    test_split = mat[f"tst{split}"][0]-1
    # test_split = [10*s+j for s in test_split for j in range(10)]
    test_split = [10*s for s in test_split]
    valid_split = mat[f"val{split}"][0]-1
    # valid_split = [10*s+j for s in valid_split for j in range(10)]
    valid_split = [10*s for s in valid_split]
    
    return train_split, test_split, valid_split

def get_dataloaders(split=1):
    strain, stest, sval = load_splits(split)
    batch_size = 8

    dataset = dataloader.FlowerDataset()

    train_dataset = Subset(dataset, strain)
    test_dataset = Subset(dataset, stest)
    valid_dataset = Subset(dataset, sval)
    # print(len(train_dataset), len(test_dataset), len(valid_dataset))
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {"train": train_dl, "test": test_dl, "valid": valid_dl}

def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return dataloader.Unnormalize(
            mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE),
            std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE),
        )(x)

class baseCAM(nn.Module):
    def __init__(self):
        super(baseCAM, self).__init__()
        base = models.vgg16(weights='DEFAULT')
        # Freeze these parameters
        for param in base.parameters():
            param.requires_grad = False


        self.features = base.features
        self.avgpool = base.avgpool

        # Make the number of feature maps equals to 17.
        self.generate_maps = nn.Sequential(
            nn.Conv2d(512, 17, 1),
            nn.ReLU(),
        ) # Output (B, 17, 7, 7)
        self.gap = nn.Sequential(
            nn.AvgPool2d((7,7)), # (B, 17, 1, 1),
            nn.Flatten(start_dim = 1), # (B, 17)
        )
        self.last_dense = nn.Linear(17, 17, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.generate_maps(x)
        x = self.gap(x)
        x = self.last_dense(x)
        return x
    
    def get_cam(self, x):
        # Return CAMs with values normalized within 0 to 1.
        B = x.shape[0]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.generate_maps(x) # Output [B, 17, 7, 7]
        # Get predictions first
        pred = self.last_dense(self.gap(x)) # [B, 17]
        pred_indices = torch.argmax(pred, dim=1)
        # Rearrange into [B, 7, 7, 17] to be able to use linear layer
        x = torch.permute(x, dims=(0,2,3,1))
        x = self.last_dense(x) # get [B, 7, 7, 17]
        x = torch.permute(x, dims=(0,3,1,2)) # [B, 17, 7, 7]
        x = x[range(x.shape[0]), pred_indices].unsqueeze(1) # [B, 1, 7, 7]
        # Normalize each map by its individual maximum
        x = x - x.flatten(start_dim=1).min(1)[0].view(-1,1,1,1)
        x = x / x.flatten(start_dim=1).max(1)[0].view(-1,1,1,1) # [B, 1, 7, 7]
        x = torchvision.transforms.Resize((256,256))(x) #[B, 1, 256, 256]
        # Use map as red channel, create zeros for green and blue channels.
        cams = torch.cat((x, torch.zeros(B, 2, 256, 256).to(DEVICE)), dim=1)
        return cams # [B, 3, 256, 256]
    
class ReCAM(baseCAM):
    def __init__(self):
        super(ReCAM, self).__init__()
        
    def recam_features(self, x):
        # Get cams then resize to agrees feature maps
        single_cams = super().get_cam(x)[:, :1, :, :] # [B, 1, 256, 256]
        single_cams = torchvision.transforms.Resize((7, 7))(single_cams) # [B, 1, 7, 7]
        # Get features
        features = self.generate_maps(self.avgpool(self.features(x))) # [B, 17, 7, 7]
        # Multiply features with cams
        features = features * single_cams # [B, 17, 7, 7]
        return features

    def forward(self, x):
        features = self.recam_features(x) # [B, 17, 7, 7]
        # Use the weighted features for classification.
        pred = self.last_dense(self.gap(features)) # [B, 17]
        return pred
    
    def get_recam(self, x):
        x = self.recam_features(x) # [B, 17, 7, 7]
        B = x.shape[0]
        # Use the weighted features for recam calculation.
        # Get predictions first
        pred = self.last_dense(self.gap(x)) # [B, 17]
        pred_indices = torch.argmax(pred, dim=1)
        # Rearrange into [B, 7, 7, 17] to be able to use linear layer
        x = torch.permute(x, dims=(0,2,3,1))
        x = self.last_dense(x) # get [B, 7, 7, 17]
        x = torch.permute(x, dims=(0,3,1,2)) # [B, 17, 7, 7]
        x = x[range(x.shape[0]), pred_indices].unsqueeze(1) # [B, 1, 7, 7]
        # Normalize each map by its individual maximum
        x = x - x.flatten(start_dim=1).min(1)[0].view(-1,1,1,1)
        x = x / x.flatten(start_dim=1).max(1)[0].view(-1,1,1,1) # [B, 1, 7, 7]
        x = torchvision.transforms.Resize((256,256))(x) #[B, 1, 256, 256]
        # Use map as red channel, create zeros for green and blue channels.
        cams = torch.cat((x, torch.zeros(B, 2, 256, 256).to(DEVICE)), dim=1)
        return cams # [B, 3, 256, 256]

    

    
def custom_save(model, path):
    """Save CAM model but only with its trainable parameters, 
    its feature extractor counterpart isn't saved.

    Args:
        model (torch.nn.Module): Trained model
        path (str): Path to model, ends with .pth.
    """
    pruned_state_dict = {
        key: item for key,item in model.state_dict().items() if not key.startswith("features.")
    }
    torch.save(pruned_state_dict, path)

def partial_load_state_dict(model, path):
    """Load a model where the state_dict from given path doesn't contains 
    all keys from model's state_dict.

    Args:
        model (torch.nn.Module): An initialized model.
        path (str): Where to load model from.
    """
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(path).items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    # Create an instance of the VGG16FeaturesOnly model
    model = ReCAM().to(DEVICE)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Load and preprocess your input image
    input_image = Image.open("./data/auged/image_0001_0.jpg")  # Replace with your input image tensor or path

    # Preprocess the input image using the same transformations as used during training
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image).to(DEVICE)
    input_batch = input_tensor.unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to obtain the intermediate feature map
    with torch.no_grad():
        features = model(input_batch)

    # Print the shape of the feature map
    # print(features.shape)
    # print(features)