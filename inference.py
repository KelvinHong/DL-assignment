import os
import argparse
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str)
    args = parser.parse_args()
    model = baseCAM().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    