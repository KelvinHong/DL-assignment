from utils import load_splits, get_dataloaders

s1, s2, s3 = load_splits(3)
print(s1.shape)


get_dataloaders(1)