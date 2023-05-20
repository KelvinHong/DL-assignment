import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import *

writer = SummaryWriter()
def train(**kwargs):
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    
    model.train()
    num_batch = len(kwargs["dataloader"])
    train_iter = iter(kwargs["dataloader"])
    epoch_loss = 0
    for i in tqdm(range(num_batch), desc=f"Training epoch {kwargs['epoch']}: "):
        bimg, blabel = next(train_iter)
        bimg, blabel = bimg.to(DEVICE), blabel.to(DEVICE)
        optimizer.zero_grad()
        prediction = model(bimg)
        loss = torch.nn.functional.cross_entropy(prediction, blabel)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), (kwargs["epoch"]-1)*num_batch + i)
    epoch_loss /= num_batch
    return epoch_loss

def eval(**kwargs):
    model = kwargs["model"]
    
    model.eval()
    num_batch = len(kwargs["dataloader"])
    train_iter = iter(kwargs["dataloader"])
    epoch_loss = 0
    for i in tqdm(range(num_batch), desc=f"Validate epoch {kwargs['epoch']}: "):
        bimg, blabel = next(train_iter)
        bimg, blabel = bimg.to(DEVICE), blabel.to(DEVICE)
        prediction = model(bimg)
        loss = torch.nn.functional.cross_entropy(prediction, blabel)
        epoch_loss += loss.item()
        writer.add_scalar('Loss/valid', loss.item(), (kwargs["epoch"]-1)*num_batch + i)
    epoch_loss /= num_batch
    return epoch_loss

if __name__ == "__main__":
    os.makedirs("./model/", exist_ok = True)
    model_path = "./model/model_test2.pth"

    model = baseCAM().to(DEVICE)
    dls = get_dataloaders(split=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 50
    best_valid_loss = float("inf")
    for epoch in range(1, epochs+1):
        train_kwargs = {
            "model": model,
            "dataloader": dls["train"],
            "optimizer": optimizer,
            "epoch": epoch,
        }
        
        train_loss = train(**train_kwargs)

        eval_kwargs = {
            "model": model,
            "dataloader": dls["valid"],
            "epoch": epoch,
        }

        valid_loss = eval(**eval_kwargs)

        print(f"Epoch {epoch}, Train vs Valid")
        print(f"\t{train_loss:.6f} vs {valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch}.")
    writer.close()