import argparse
import os
from tqdm import tqdm
from copy import copy
import json
from utils import *

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def evaluate(model):
    # Given a model, load weights from model_path and perform evaluation.
    data = get_dataloaders()["valid"]
    total_data = len(data.dataset)
    num_batch = len(data)
    train_iter = iter(data)
    accuracies = {
        "top1": 0,
        "top3": 0,
        "top5": 0,
    }
    for i in tqdm(range(num_batch), desc=f"Evaluating: "):
        bimg, blabel = next(train_iter)
        bimg, blabel = bimg.to(DEVICE), blabel.to(DEVICE)
        B = blabel.shape[0]
        prediction = model(bimg)
        top1, top3, top5 = accuracy(prediction, blabel, topk = (1,3,5))
        accuracies["top1"] += int(top1.item() * B)
        accuracies["top3"] += int(top3.item() * B)
        accuracies["top5"] += int(top5.item() * B)
    accuracies = {key: round(value/total_data, 4) for key, value in accuracies.items()}
    return accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--model-dir", type=str)
    # Model type will be inferred by training logs
    args = parser.parse_args()
    model_dir = args.model_dir
    # Infer model type
    model = None
    with open(os.path.join(model_dir, "training_logs.txt"), "r") as f:
        first_line = f.readline()
        print(first_line)
        if "[CAM]" in first_line:
            model = baseCAM()
        elif "[ReCAM]" in first_line:
            if "[sigmoid]" in first_line:
                model = ReCAM("sigmoid")
            elif "[relu]" in first_line:
                model = ReCAM("relu")
            elif "[minmax]" in first_line:
                model = ReCAM("minmax")
    model.eval()
    model.to(DEVICE)
    # Initialize json file
    eval_out = os.path.join(model_dir, "eval.json")
    # Start evaluating
    evals = {}
    for ind in range(100):
        model_path = os.path.join(model_dir, f"epoch_{ind+1}.pth")
        if not os.path.isfile(model_path):
            continue
        print(f"On model from epoch {ind+1}...")
        model = partial_load_state_dict(model, model_path)
        accuracies = evaluate(model)
        evals[f"epoch_{ind+1}"] = copy(accuracies)
    with open(eval_out, "w") as f:
        json.dump(evals, f, indent=4)
    