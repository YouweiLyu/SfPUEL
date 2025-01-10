import os
import cv2
import torchvision
import numpy as np
import glob
from torch.utils import data

from . import loader_utils as util

class normal_dataloader(data.Dataset):
    def __init__(self, args, path, split='val'):
        super().__init__()
        self.path = path
        self.split = split

        self.grayscale = args.grayscale
        
        self.files = []
        self.transform = torchvision.transforms.ToTensor()

        self.load_normal = 'normal_gt' in args.loaded_data_names
        self.load_material  = 'material_label_gt' in args.loaded_data_names

        pol000_path = os.path.join(path, 'pol000')
        pol045_path = os.path.join(path, 'pol045')
        pol090_path = os.path.join(path, 'pol090')
        pol135_path = os.path.join(path, 'pol135')
        material_path = os.path.join(path, 'material_tag')
        m_path = os.path.join(path, 'mask')
        n_path = os.path.join(path, args.in_normal_dir)
        pol000_paths = glob.glob(f'{path}/pol000/*.png')
        for path in sorted(pol000_paths):
            filename = os.path.basename(path)
            id_name = filename[:-4]
            file_dict = {
                'pol000': os.path.join(pol000_path, filename),
                'pol045': os.path.join(pol045_path, filename),
                'pol090': os.path.join(pol090_path, filename),
                'pol135': os.path.join(pol135_path, filename),
                'mask': os.path.join(m_path, filename),
                'normal': os.path.join(n_path, filename),
                'material': os.path.join(material_path, filename),
                'name': id_name,
            }
            self.files.append(file_dict)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        pol000 = util.read_img_rgb(datafiles['pol000'])
        pol090 = util.read_img_rgb(datafiles['pol090'])
        pol045 = util.read_img_rgb(datafiles['pol045'])
        pol135 = util.read_img_rgb(datafiles['pol135'])
        if self.grayscale:
            pol000 = cv2.cvtColor(pol000, cv2.COLOR_RGB2GRAY)[..., None]
            pol090 = cv2.cvtColor(pol090, cv2.COLOR_RGB2GRAY)[..., None]
            pol045 = cv2.cvtColor(pol045, cv2.COLOR_RGB2GRAY)[..., None]
            pol135 = cv2.cvtColor(pol135, cv2.COLOR_RGB2GRAY)[..., None]
        
        polars = np.stack([pol000, pol045, pol090, pol135], 0)
        h, w = pol000.shape[0], pol000.shape[1]
        if os.path.isfile(datafiles['mask']):
            mask = util.read_mask(datafiles['mask'])
            mask_flag = True
        else:
            mask = np.ones((h, w, 1), dtype=np.float32)
            mask_flag = False
        if self.load_normal:
            normal = util.read_img_rgb_norm(datafiles['normal'])
        
        if self.load_material:
            material = util.read_img_rgb_norm(datafiles['material'])
            material = material[...,1]>0.5
        
        if self.split in ['test', 'val']:
            patch_size = 512
            mask, polars, roi = util.crop_augment(mask, polars, patch_size=patch_size, mask_flag=mask_flag)
            polars = util.normalize_img(polars)
            polars = util.normalize_augment(polars, mask)
        else:
            polars = util.normalize_img(polars)
            polars = util.normalize_augment(polars, mask)
    
        mask = self.transform(mask)

        sample = {'polar':polars, 'mask':mask, 'name': datafiles['name']}
        if self.split in ['test', 'val']:
            sample['roi'] = roi
        if self.load_normal:
            normal = self.transform(normal)
            sample['normal_gt'] = normal
        if self.load_material:
            sample['material_label_gt'] = material
        return sample
