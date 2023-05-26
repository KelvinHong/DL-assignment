import os
from tqdm import tqdm
import datetime
import argparse
import sys # Used for redirect model training output
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

def CAM_workflow(model_dir: str, normalize_by="minmax"):
    model = baseCAM(normalize_by=normalize_by).to(DEVICE)
    dls = get_dataloaders(split=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
            custom_save(model, os.path.join(model_dir, f"epoch_{epoch}.pth"))
            print(f"Model saved at epoch {epoch}.")
    writer.close()

def ReCAM_workflow(model_dir: str, normalize_by="minmax"):
    # Following this https://arxiv.org/pdf/2203.00962.pdf
    model = ReCAM(normalize_by="minmax").to(DEVICE)
    dls = get_dataloaders(split=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
            custom_save(model, os.path.join(model_dir, f"epoch_{epoch}.pth"))
            print(f"Model saved at epoch {epoch}.")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", help="Choose from 'CAM' or 'ReCAM'", type=str)
    parser.add_argument("-n", "--normalize-by", help="Choose from 'minmax', 'sigmoid' or 'relu', defaults to 'minmax'.", type=str, default='minmax')
    args = parser.parse_args()
    os.makedirs("./model/", exist_ok = True)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    normalize_by = args.normalize_by
    model_type =  args.model_type

    # No need to touch below
    model_dir = f"./model/TS{timestamp}_{model_type}/"
    os.makedirs(model_dir)
    
    filename = os.path.join(model_dir, "training_logs.txt")
    # Record model meta 
    with open(filename, "w") as f:
        f.write(f"Model type: [{model_type}], using [{normalize_by}] normalizing method.\n")
    # Redirect output to the file
    sys.stdout = open(filename, "a")
    if model_type == "CAM":
        CAM_workflow(model_dir, normalize_by=normalize_by)
    elif model_type == "ReCAM":
        ReCAM_workflow(model_dir, normalize_by=normalize_by)
    # Restore output to default.
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    