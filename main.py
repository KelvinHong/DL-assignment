import os
from tqdm import tqdm
import datetime
import argparse
import sys # Used for redirect model training output
from torch.utils.tensorboard import SummaryWriter
import json
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
    valid_iter = iter(kwargs["dataloader"])
    epoch_loss = 0
    for i in tqdm(range(num_batch), desc=f"Validate epoch {kwargs['epoch']}: "):
        bimg, blabel = next(valid_iter)
        bimg, blabel = bimg.to(DEVICE), blabel.to(DEVICE)
        prediction = model(bimg)
        loss = torch.nn.functional.cross_entropy(prediction, blabel)
        epoch_loss += loss.item()
        writer.add_scalar('Loss/valid', loss.item(), (kwargs["epoch"]-1)*num_batch + i)
    epoch_loss /= num_batch
    return epoch_loss

def CAM_workflow(model_dir: str, **kwargs):
    model = baseCAM(normalize_by=kwargs["normalize_by"]).to(DEVICE)
    dls = get_dataloaders(split=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
    epochs = kwargs["epochs"]
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
        if valid_loss > 2 * train_loss:
            print("Model stopped training as reaching overfitting: Valid_loss is more than twice of training loss.")
            break
    writer.close()

def ReCAM_workflow(model_dir: str, **kwargs):
    # Following this https://arxiv.org/pdf/2203.00962.pdf
    model = ReCAM(normalize_by=kwargs["normalize_by"]).to(DEVICE)
    dls = get_dataloaders(split=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
    epochs = kwargs["epochs"]
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
        if valid_loss > 2 * train_loss:
            print("Model stopped training as reaching overfitting: Valid_loss is more than twice of training loss.")
            break
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", help="Choose from 'CAM', 'ReCAM' or 'LayerCAM", type=str)
    parser.add_argument("-n", "--normalize-by", help="Choose from 'minmax', 'sigmoid' or 'relu', defaults to 'minmax'.", type=str, default='minmax')
    parser.add_argument("-l", "--lr", help="Learning rate, defaults to 1e-4.", default=1e-4, type=float)
    parser.add_argument("-e", "--epochs", help="Number of epochs, defaults to 50.", default=50, type=int)
    args = parser.parse_args()
    os.makedirs("./model/", exist_ok = True)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    normalize_by = args.normalize_by
    model_type =  args.model_type

    # No need to touch below
    model_dir = f"./model/TS{timestamp}_{model_type}/"
    os.makedirs(model_dir)

    training_kwargs = {
        "epochs": args.epochs,
        "lr": args.lr,
        "normalize_by": args.normalize_by,
    }
    # Save keywords
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        f.write(json.dumps(training_kwargs, indent=4))
    
    filename = os.path.join(model_dir, "training_logs.txt")
    # Record model meta 
    with open(filename, "w") as f:
        f.write(f"Model type: [{model_type}], using [{normalize_by}] normalizing method.\n")
    # Redirect output to the file
    sys.stdout = open(filename, "a")
    if model_type == "CAM":
        CAM_workflow(model_dir, **training_kwargs)
    elif model_type == "ReCAM":
        ReCAM_workflow(model_dir, **training_kwargs)
    # Restore output to default.
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    