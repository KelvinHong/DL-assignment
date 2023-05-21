from inference import get_random_batch
from dataloader import FlowerDataset
d = FlowerDataset()
batch, labels = get_random_batch(d, seed=1000)
print(batch.shape)
print(batch.min())
print(batch.max())
print(labels)