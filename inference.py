"""Labels:
0 - daffodil (0001 - 0080)
1 - snowdrop (0081 - 0160)
2 - lily_valley (0161 - 0240)
3 - bluebell (0241 - 0320)
4 - crocus (0321 - 0400)
5 - iris (0401 - 0480)
6 - tigerlily (0481 - 0560)
7 - tulip (0561 - 0640)
8 - fritillary (0641 - 0720)
9 - sunflower (0721 - 0800)
10 - daisy (0801 - 0880)
11 - colts_foot (0881 - 0960)
12 - dandelion (0961 - 1040)
13 - cowslip (1041 - 1120)
14 - buttercup (1121 - 1200)
15 - windflower (1201 - 1280)
16 - pansy (1281 - 1360)
"""

import os
import argparse
from numpy.random import default_rng
import matplotlib.pyplot as plt
import datetime
from dataloader import FlowerDataset
from utils import *

NROW, NCOL = 3,4
def get_random_batch(dataset, n=NROW * NCOL, seed = -1):
    total = len(dataset)
    if seed == -1:
        rng = default_rng()
    else:
        rng = default_rng(int(seed))
    indices = rng.choice(list(range(total)), size=n, replace=False)
    selected = [dataset[i] for i in indices]
    batched_tensor = torch.stack([data[0] for data in selected])
    labels = [data[1] for data in selected]
    return batched_tensor, labels

def create_comparison(gts, predictions):
    # gts should be a list [B] (integers), 
    # predictions should be a torch tensor [B, 17] (float tensor)
    result = {}
    # Create captions
    captions = []
    for gt, pred in zip(gts, predictions):
        gt_label = LABELS[gt]
        caption = f"Ground truth [{gt}-{gt_label}]\n"
        top3 = torch.topk(pred, 3)[1]
        top3_labels = [LABELS[ind] for ind in top3]
        caption += f"Prediction Top3 [{', '.join(top3_labels)}]"
        captions.append(caption)
    result["captions"] = captions
    # Calculate Cross Entropy Loss
    loss = torch.nn.functional.cross_entropy(predictions, torch.tensor(gts).to(DEVICE)).item()
    result["loss"] = loss
    return result

def save_as_grids(batched_tensors, info, save_as):
    """Save tensors into a single image.

    Args:
        batched_tensors (torch tensor): A torch tensor with shape [B, C, H, W]
        labels (list or torch tensor): Labels with shape [B].
        save_as (str): Save as path.
    """
    batched_tensors = batched_tensors.detach().cpu().numpy().transpose((0,2,3,1))
    fig = plt.figure()
    fig.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    fig.suptitle(f"Loss evaluated as {info['loss']:.4f}", fontsize=16)
    for ind in range(NROW*NCOL):
        ax = plt.subplot(NROW, NCOL, ind+1)
        ax.set_title(info["captions"][ind], size=10)
        ax.imshow(batched_tensors[ind])
    fig.set_size_inches(16, 14)
    fig.savefig(save_as, dpi=120)

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    image_output = os.path.join("./output/", f"grid_{timestamp}.jpg")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str)
    parser.add_argument("-s", "--seed", type=int, default=-1)
    args = parser.parse_args()
    # Only use valid dataset
    d = FlowerDataset()
    strain, stest, sval = load_splits(split=1)
    valid_dataset = Subset(d, sval)
    # Start
    model = baseCAM().to(DEVICE)
    model = partial_load_state_dict(model, args.model_path).to(DEVICE)
    model.eval()
    batch_input, batch_label = get_random_batch(valid_dataset, seed=args.seed)
    batch_input = batch_input.to(DEVICE)
    predictions = model(batch_input)
    # Create comparison labels 
    info = create_comparison(batch_label, predictions)
    # Show grid
    save_as_grids(batch_input, info, image_output)

    # Test CAM
    model.get_cam(batch_input)

    