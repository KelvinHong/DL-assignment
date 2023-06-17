import requests

print("Downloading dataset tar file, please wait for a few seconds...")
url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
target_path = './data/17flowers.tgz'

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(target_path, 'wb') as f:
        f.write(response.raw.read())
    print("File downloaded. ")
else:
    print("File couldn't be downloaded, please download it " +\
        "from https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz" +\
        "manually and put it in the \"data/\" folder")


