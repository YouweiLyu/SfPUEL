import os
import cv2
import numpy as np

def ddepth2type(ddepth):
    if ddepth == 65535:
        dtype = np.uint16
    elif ddepth == 255:
        dtype = np.uint8
    return dtype

def unify_shape(gt, img, split='img'):
    h_gt, w_gt = gt.shape[:2]
    if img.shape[0]!=h_gt or img.shape[1]!=w_gt:
        if split.lower() == 'img':
            img = cv2.resize(img, (w_gt, h_gt), cv2.INTER_LINEAR)
        elif split.lower() == 'normal':
            img = cv2.resize(img, (w_gt, h_gt), cv2.INTER_LINEAR)
            norm = np.sqrt((img**2).sum(axis=-1, keepdims=True))
            img = img / (norm + 1e-6)
        elif split.lower() == 'mask':
            img = cv2.resize(img, (w_gt, h_gt), cv2.INTER_NEAREST)
        else:
            raise ValueError('Invalid type of input img')
    else:
        return img
    return img

def check_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    return path

def make_folders(f_list):
    for f in f_list:
        check_dir(f)

def save_normal(normal, mask, path, ddepth=65535):
    n = ((normal + 1) / 2. * mask).clip(0, 1)
    n = n[..., ::-1] * ddepth
    cv2.imwrite(path, n.astype(ddepth2type(ddepth)))

def dict2string(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        if v or v == 0:
            strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs
