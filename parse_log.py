# Copyright 2023 user
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script parse the log files to extract training results into structured data
import os
import matplotlib.pyplot as plt

def parse_log_to_loss(model_dir):
    log_file = os.path.join(model_dir, "training_logs.txt")
    if not os.path.isfile(log_file):
        raise ValueError("Couldn't locate log file. Please try again.")
    with open(log_file, "r") as f:
        lines = f.read().splitlines()
    # Collecting losses
    train_losses = []
    valid_losses = []
    cue = False # cueing next line to store losses
    for line in lines:
        if cue: 
            train_loss, _, valid_loss = line.split()
            train_losses.append(float(train_loss))
            valid_losses.append(float(valid_loss))
            cue = False
        elif line.startswith("Epoch "):
            cue = True

    l = len(train_losses)
    # Detect overfitting epoch
    overfit_ind = None
    for i in range(l):
        if train_losses[i] * 2 < valid_losses[i]:
            overfit_ind = i
            break

    # Plot
    plt.clf()
    plt.plot(range(1, l+1), train_losses, label="Train Loss", color="g")
    plt.plot(range(1, l+1), valid_losses, label="Valid Loss", color="y")
    if overfit_ind is not None:
        plt.axvline(x = overfit_ind+1, color = 'r', label = 'Overfitting from here')
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.title(f'Training Losses for [{model_dir}].')
    # plt.show()
    plt.savefig(os.path.join(model_dir, "performance_graph.png"))

if __name__ == "__main__":
    for subdir in os.listdir("./model/"):
        # print(subdir)
        parse_log_to_loss(f"model/{subdir}/")