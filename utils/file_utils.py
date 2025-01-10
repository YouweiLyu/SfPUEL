import os
import cv2
import numpy as np

def get_filename(pred_path, method):
    filename = os.path.basename(pred_path)
    filename = filename.split(f'_{method}_')[0]
    return filename

def get_normal_gt_path(filename, gt_dir):
    gt_path = os.path.join(gt_dir, 'normal', filename+'.png')
    return gt_path

def get_mask_path(filename, gt_dir):
    mask_path = os.path.join(gt_dir, 'mask', filename+'.png')
    return mask_path

def read_img(path):
    img = cv2.imread(path, -1)
    depth = 65535. if img.dtype == np.uint16 else 255.
    img_norm = img.astype(np.float32) / depth
    return img_norm

def read_normal(path):
    ext = path.split('.')[-1]
    if ext in ['png', 'jpg', 'tiff']:
        n_img = read_img(path)
        normal = n_img * 2 -1
    elif path.endswith('.npy'):
        normal = np.load(path)
    else:
        raise ValueError(f'Invalid file extension')
    # norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.sqrt((normal**2).sum(axis=-1, keepdims=True))
    normal = normal / (norm + 1e-8)
    # normal = 
    return normal

def read_mask(path):
    if os.path.isfile(path):
        ext = path.split('.')[-1]
        if ext in ['png', 'jpg', 'tiff']:
            mask = read_img(path)
        elif path.endswith('.npy'):
            mask = np.load(path)
        else:
            raise ValueError(f'Invalid file extension')
        mask = (mask > 0.5).astype(bool)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
    else:
        print(f'Mask file "{path}" doesnot exsit')
        mask = None
    return mask
