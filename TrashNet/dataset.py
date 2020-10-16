import torch as t 
from torch.utils import data 
from torchvision import transforms as T 
from PIL import Image  
from torchvision.datasets import ImageFolder
import random
import os

# trash_dataset = ImageFolder('datasets/dataset-resized', transform = transforms)
# trash_dataset.class_to_idx
# # {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
# trash_dataset[0][0] #(512, 384)
# trash_dataset[0][1]

# ImageFolder无法划分训练集和验证集，并对其进行不同的transform，除非有两个dir分别存放训练数据和验证数据
# 故只能用最原始的data.Dataset方法

# imgs = ImageFolder(self.root)
# index = list(range(len(imgs)))
# random.seed(self.seed)
# random.shuffle(index)
# num_train = int(0.7 * len(imgs))
# if self.train:
#     # training set
#     dataset = data.Subset(imgs, index[:num_train])
# else:
#     # validation set
#     dataset = data.Subset(imgs, index[num_train:])

# normalize = T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
# transforms = T.Compose([
#     T.Scale(256),
#     T.RandomSizedCrop(224),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     normalize
# ])

# count = {}
# for i in range(len(trash_dataset)):
#     label = trash_dataset[i][1]
#     if label in count.keys():
#         count[label] += 1
#     else:
#         count[label] = 1
# count


class Trash(data.Dataset):
    def __init__(self, root, transform = None, train = True, seed = 1):
        self.labels = {'paper':0, 'glass':1, 'plastic':2, 'cardboard':3, 'trash':4, 'metal':5}
        dirs = os.listdir(root)
        imgs = []
        for d in dirs:
            path = os.path.join(root, d)
            for img in os.listdir(path):
                imgs.append(os.path.join(path, img))

        random.seed(seed)
        random.shuffle(imgs)
        num_train = int(0.7 * len(imgs))
        if train:
            # training
            self.imgs = imgs[:num_train]
        else:
            # validation
            self.imgs = imgs[num_train:]

        if transform is None:    
            normalize = T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
            if not train:
                # 验证集
                self.transform = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                # 训练集，做数据增广
                self.transform = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[img_path.split('/')[-2]]
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
        

# trashdata = Trash('datasets/dataset-resized')
# T.ToPILImage()(trashdata.__getitem__(200)[0] * 0.5 + 0.5)
# trashdata.__getitem__(200)[1]



    
