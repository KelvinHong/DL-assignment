import requests
import os
import tarfile
import gdown

def download_file(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.raw.read())
        print(f"File downloaded to {path} . ")

data_dir = "./data/"
os.makedirs(data_dir, exist_ok=True)

# Download split 
split_path = os.path.join(data_dir, "datasplits.mat")
download_file("https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat", 
              split_path)

print("Downloading dataset tar file, please wait for a few seconds...")
target_path = os.path.join(data_dir,'17flowers.tgz')
download_file('https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
              target_path)

# Decompress tar file.
tar = tarfile.open(target_path, "r")
tar.extractall(path=data_dir)
tar.close()

# Remove tgz file
os.remove(target_path)

# Download models
import gdown

url = 'https://drive.google.com/uc?id=1Trr1q4FzWSNfeVq47RhudJC_lLlxL8J9'
model_rarpath = './models.rar'
gdown.download(url, model_rarpath, quiet=False)