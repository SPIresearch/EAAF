import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [
                (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
                for val in image_list
            ]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def l_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("L")


class ImageList(Dataset):
    def __init__(
        self, image_list, labels=None, transform=None, target_transform=None, mode="RGB"
    ):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_aug(Dataset):
    def __init__(
        self,
        image_list,
        labels=None,
        transform=None,
        transform1=None,
        transform2=None,
        mode="RGB",
    ):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders"))

        self.imgs = imgs
        self.transform = transform
        self.transform1 = transform1

        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img1 = np.expand_dims(np.array(img), 0)

        if self.transform is not None:
            img1 = self.transform(images=img1)

        img1 = np.array(img1, dtype=np.float32)
        img1 = torch.from_numpy(img1)[0]
        img1 = img1.permute(2, 0, 1)

        img1 = self.transform1(img1)

        return img1, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx_aug(Dataset):
    def __init__(
        self,
        image_list,
        labels=None,
        transform=None,
        transform1=None,
        transform2=None,
        mode="RGB",
    ):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders"))

        self.imgs = imgs
        self.transform = transform
        self.transform1 = transform1

        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img1 = np.expand_dims(np.array(img), 0)

        if self.transform is not None:
            img1 = self.transform(images=img1)

        img1 = np.array(img1, dtype=np.float32)
        img1 = torch.from_numpy(img1)[0]
        img1 = img1.permute(2, 0, 1)

        img1 = self.transform1(img1)

        return img1, target, index

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(
        self,
        image_list,
        labels=None,
        transform=None,
        target_transform=None,
        transform1=None,
        mode="RGB",
    ):
        imgs = make_dataset(image_list, labels)
    

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.transform1 = transform1
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img1 = self.transform(img)
        if self.transform1 is not None:
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform1 is not None:
            return [img1, img2], target, index
        else:
            return img1, target, index

    def __len__(self):
        return len(self.imgs)


class Listset2(Dataset):
    def __init__(self, parent_dataset1, parent_dataset2, lists, uns=False):

        self.parent_dataset1 = parent_dataset1
        self.parent_dataset2 = parent_dataset2
        self.lists = lists
        self.uns = uns

    def __getitem__(self, index):
        ids = self.lists[index]
        x1, _, tar_1 = self.parent_dataset1.__getitem__(ids[0])
        x2, _, tar_2 = self.parent_dataset2.__getitem__(ids[1])
        img = ids[2] * x1 + (1 - ids[2]) * x2
        return img, tar_1, tar_2

    def __len__(self):
        return len(self.lists)


class Listset(Dataset):
    def __init__(self, parent_dataset, lists, uns=False):

        self.parent_dataset = parent_dataset
        self.lists = lists
        self.uns = uns

    def __getitem__(self, index):
        ids = self.lists[index]
        if self.uns:
            return self.parent_dataset.__getitem__(ids), index
        else:
            return self.parent_dataset.__getitem__(ids)

    def __len__(self):
        return len(self.lists)


class Listset3(Dataset):
    def __init__(self, parent_dataset, lists, label):

        self.parent_dataset = parent_dataset
        self.lists = lists
        self.label = label

    def __getitem__(self, index):
        ids = self.lists[index]

        return self.parent_dataset.__getitem__(ids), self.label[index]

    def __len__(self):
        return len(self.lists)
