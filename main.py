import os
from utils import *



if __name__ == "__main__":
    os.makedirs("./model/", exist_ok = True)
    model_path = "./model/model_test1.pth"

    model = baseCAM()
    dls = get_dataloaders(split=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = iter(dls["train"])
    bimg, blabel = next(train_iter)
    model.eval()
    output = model(bimg)
    loss = torch.nn.functional.cross_entropy(output, blabel)
    print(loss)