from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange
# -*- coding: utf-8 -*-
def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class Whole_Slide_Bag(Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


from skimage.measure import regionprops
def extract_features(mask, prob_map):
    props = regionprops(mask, prob_map)  # 使用区域属性分析工具
    features = []
    for prop in props:
        features.append({
            'coordinate_x': prop.centroid[1],
            'coordinate_y': prop.centroid[0],
            'cell_type': 1,  # 设为1表示肿瘤细胞
            'probability': np.mean(prob_map[prop.coords[:, 0], prop.coords[:, 1]]),
            'area': prop.area,
            'convex_area': prop.convex_area,
            'eccentricity': prop.eccentricity,
            'extent': prop.extent,
            'filled_area': prop.filled_area,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'orientation': prop.orientation,
            'perimeter': prop.perimeter,
            'solidity': prop.solidity,
            'pa_ratio': prop.major_axis_length / prop.minor_axis_length
        })
    return features
class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')



        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchstain
# from dataset_modules.dataset_h5 import eval_transforms
T_all=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
class Whole_Slide_Bag_FP_new(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1,
                 stain_method: str = None,
                 stain_target_path: str = None
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi: OpenSlide WSI object
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
            stain_method (str): 'reinhard' | 'macenko' | 'vahadane'
            stain_target_path (str): Path to target image for stain normalizer
        """
        self.pretrained = pretrained
        self.wsi = wsi
        # transforms
        self.roi_transforms = custom_transforms if custom_transforms else eval_transforms(pretrained=pretrained)
        # stain normalizer
        self.stain_method = stain_method.lower() if stain_method else None


        if self.stain_method and stain_target_path:
            target = Image.open(stain_target_path).convert('RGB')
            self.stain_target = np.array(target)
            if self.stain_method == 'reinhard':
                self.stainer = torchstain.normalizers.ReinhardNormalizer()
            elif self.stain_method == 'macenko':
                self.stainer = torchstain.normalizers.MacenkoNormalizer(backend='torch')#torchstain.MacenkoNormalizer()
            elif self.stain_method == 'vahadane':
                self.stainer = torchstain.normalizers.VahadaneNormalizer()
            else:
                raise ValueError(f"Unknown stain_method {stain_method}")
            self.stainer.fit(T_all(self.stain_target))
            # normalizer.fit(T(target))
        else:
            self.stainer = None
        # h5 params
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        print(f"Dataset length: {self.length}")
        print(f"Patch level: {self.patch_level}, Patch size: {self.patch_size}")
        print(f"Target patch size: {self.target_patch_size}")
        print(f"Pretrained: {self.pretrained}")
        print(f"Stain method: {self.stain_method}")

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        from histaugan.options import TestOptions
        from histaugan.model import MD_multi
        parser = TestOptions()
        opts = parser.parse()
        img1 = img
        # model
        print('\n--- load model ---')
        model = MD_multi(opts)
        model.setgpu(opts.gpu)
        model.resume(opts.resume, train=False)
        model.eval()
        arr = np.array(img)
        arr1 = Image.fromarray(arr)
        arr1.save(os.path.join("M:\\STAS_2025_buchong\\xiangya2_feature", str(idx) + "yuanshi" + '.png'))
        img = T_all(arr)#.cuda()
        img = img.unsqueeze(0).cuda()
        # model = model.cuda()
        imgs = []
        a = model.test_forward_random(img)
        for i in range(1,opts.num_domains):
            imgs.append(a[i])
        from histaugan.saver import tensor2img
        j=0
        for img2 in zip(imgs):
            img2 = tensor2img(img2)
            img2 = Image.fromarray(img2)
            img2.save(os.path.join("M:\\STAS_2025_buchong\\xiangya2_feature", str(idx) + str(j) +'.png'))
            j = j+1

        # apply stain normalization
        if self.stainer:
            arr = np.array(img1)
            to_transform = T_all(arr)
            image, H, E = self.stainer.normalize(I=to_transform, stains=True)
            # 步骤1: 调整维度顺序为 (3, 256, 256) 以符合PyTorch的通道优先格式
            image = image.permute(2, 0, 1).to(torch.uint8)
            # 将张量转换为 PIL 图像以应用转换
            img = transforms.ToPILImage()(image)

            image1 = np.array(img)
            # image1 = np.transpose(image1, (1, 2, 0))
            image1 = Image.fromarray(image1)
            image1.save(os.path.join("M:\\STAS_2025_buchong\\xiangya2_feature", str(idx) + "normal" + '.png'))

        # resize
        if self.target_patch_size:
            img = img.resize(self.target_patch_size, Image.BILINEAR)
        # to tensor
        img_tensor = self.roi_transforms(img).unsqueeze(0)
        return img_tensor, coord














class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path,encoding='gbk')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]




