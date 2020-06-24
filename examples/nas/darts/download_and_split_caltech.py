from PIL import Image
import os
import os.path
import numpy as np
from sklearn.model_selection import train_test_split

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

from datasets import Cutout
print('imports working')

def download(root):


    download_and_extract_archive(
        "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
        root,
        filename="101_ObjectCategories.tar.gz",
        md5="b224c7392d521a49829488ab0f1120d9")
    download_and_extract_archive(
        "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
        root,
        filename="101_Annotations.tar",
        md5="6f83eeb1f24d99cab4eb377263132c91")

# download('data')
original_root = 'data/caltech/101_ObjectCategories/101_ObjectCategories'
train_path = 'data/train'
val_path = 'data/val'
test_path = 'data/test'
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
cats = os.listdir(original_root)
mega_cats = []
mega_img = []
for i in cats:
    fold_path = os.path.join(original_root,i)
    for k in os.listdir(fold_path):
        mega_cats.append([i])
        mega_img.append([os.path.join(fold_path,k)])

cat = np.array(mega_cats)
img = np.array(mega_img)
Itrain, img_test, Ctrain, cat_test = train_test_split(img, cat, test_size=0.2, random_state=42)
img_train, img_val, cat_train, cat_val = train_test_split(Itrain, Ctrain, test_size=0.2, random_state=42)
img_train =  img_train.tolist()
cat_train =cat_train.tolist()
img_val =img_val.tolist()
cat_val =cat_val.tolist()
img_test =img_test.tolist()
cat_test =cat_test.tolist()
## move train
for i in range(0,len(img_train)):
    print(cat_train[i][0])
    print(img_train[i][0])
    os.makedirs(os.path.join(train_path,cat_train[i][0]), exist_ok=True)
    img = Image.open(img_train[i][0])
    if len(img.getbands()) <3:
        continue
    img_name = img_train[i][0].split('\\')[-1]
    print(img_name)
    if img_name.split('.')[-1] == 'jpg':
        l = len(os.listdir(os.path.join(train_path,cat_train[i][0])))
        img.save(os.path.join(train_path,cat_train[i][0])+'/'+"image_{:04d}.jpg".format(l+1))

## move validation
for i in range(0,len(img_val)):
    print(cat_val[i][0])
    print(img_val[i][0])
    os.makedirs(os.path.join(val_path,cat_val[i][0]), exist_ok=True)
    img = Image.open(img_val[i][0])
    if len(img.getbands()) <3:
        continue
    img_name = img_val[i][0].split('\\')[-1]
    print(img_name)
    if img_name.split('.')[-1] == 'jpg':
        l = len(os.listdir(os.path.join(val_path,cat_val[i][0])))
        img.save(os.path.join(val_path,cat_val[i][0])+'/'+"image_{:04d}.jpg".format(l+1))


## move test
for i in range(0,len(img_test)):
    print(cat_test[i][0])
    print(img_test[i][0])
    os.makedirs(os.path.join(test_path,cat_test[i][0]), exist_ok=True)
    img = Image.open(img_test[i][0])
    if len(img.getbands()) <3:
        continue
    img_name = img_test[i][0].split('\\')[-1]
    print(img_name)
    if img_name.split('.')[-1] == 'jpg':
        l = len(os.listdir(os.path.join(test_path,cat_test[i][0])))
        img.save(os.path.join(test_path,cat_test[i][0])+'/'+"image_{:04d}.jpg".format(l+1))
