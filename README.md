# Visiting some Class Activation Map (CAM) models

This repository includes the codes for the study of CAM models, 
such model can visualize regions of input image that activates the 
decision of the model, which is typically used to interpret 
the inner working of an image classifier. 

## Pre-requisite

This program is tested on Python 3.9.0, but should work on later version
of python. It is also recommended to run this in a device with GPU as
it is not tested on a CPU-only machine.

If you do not have virtual environment module, install it by
```
pip install virtualenv
```
Then create a virtual environment by
```
virtualenv venv
```
A folder `./venv/` should appear. Then, activate venv by (assuming you're using 
Windows 10)
```
./venv/Scripts/activate
```

Install `torch` and `torchvision` (with GPU recommended)
from the official pytorch website. 
GPU example: 
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

Install requirements by running
```
pip install -r requirements.txt
```

## Steps to use this repository. 

(Must do)

Run `python prepare.py` to download dataset, splits and models. 

Decompress pretrained models from `./models.rar` so that `/model/` 
appear. 

Perform data augmentation by running `python augment.py`.


(Optional)

To train a model, use `main.py` (need to understand flags)

To inference, use `inference.py` (need to understand flags)

To evaluate (accuracies), use `evaluate.py` (need to understand flags)


## Results

![Comparison of three models](./media/compare_98.png)