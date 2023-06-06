# from utils import *
import os
for ind, directory in enumerate(os.listdir("./model/")):
    full = os.path.join("./model/", directory)
    print("="*10 + f"Evaluating model [{ind+1}: {directory}]" + "="*10)
    os.system(f"python evaluate.py -d {full}")