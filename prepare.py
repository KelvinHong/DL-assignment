import requests
import os
import tarfile

data_dir = "./data/"
os.makedirs(data_dir, exist_ok=True)

# Download split 
split_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat"
split_path = os.path.join(data_dir, "datasplits.mat")
response = requests.get(split_url, stream=True)
if response.status_code == 200:
    with open(split_path, 'wb') as f:
        f.write(response.raw.read())
    print("Split file downloaded. ")

print("Downloading dataset tar file, please wait for a few seconds...")
url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
target_path = os.path.join(data_dir,'17flowers.tgz')

# Download tar file
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(target_path, 'wb') as f:
        f.write(response.raw.read())
    print("Dataset file downloaded. ")
else:
    print("File couldn't be downloaded, please download it " +\
        "from https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz" +\
        "manually and put it in the \"data/\" folder")
    exit()

# Decompress tar file.
tar = tarfile.open(target_path, "r")
tar.extractall(path=data_dir)
tar.close()

# Remove tgz file
os.remove(target_path)