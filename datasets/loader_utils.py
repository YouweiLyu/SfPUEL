import cv2
import numpy as np

type_depth_dict = {np.uint8: 255., np.uint16: 65535.}

def normalize_img(img):
    try:
        depth = 65535. if img.dtype == np.uint16 else 255.
    except:
        raise OSError(f'Failed to load the image')
    img_norm = img.astype(np.float32) / depth
    return img_norm

def read_img_rgb(path):
    img = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    return img

def read_img_rgb_norm(path):
    img = read_img_rgb(path)
    img = normalize_img(img)
    return img

def read_img_cv2_norm(path):
    img = cv2.imread(path, -1)
    img = normalize_img(img)
    return img

def read_mask(path):
    """Get mask ndarray from the path
    
    Returns: 
        Mask ndarray with shape of (H,W,1) in float32 format
    Raises:
        OSError: fail to load mask file
    """
    try:
        mask = (cv2.imread(path, -1) > 0.5).astype(np.float32)
    except:
        raise OSError(f'Failed to load "{path}"')
    if len(mask.shape) == 3:
        mask = mask[...,:1]
    elif len(mask.shape) == 2:
        mask = mask[...,None]
    else:
        raise Exception()
    return mask

def read_normal(path):
    """Get normal ndarray from the path
    Returns: 
        Normal ndarray ranging (-1, 1) in float32 format
    Raises:
        OSError: fail to load mask file and unsupported normal file type
    """
    file_ext = path.split('.')[-1]
    if file_ext.lower() in ['png', 'jpg', 'tiff', 'tif', 'bmp']:
        try:
            normal_img = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
        except:
            raise OSError(f'Failed to load "{path}"')
        normal = normalize_img(normal_img) * 2 - 1
    elif file_ext.lower() == '.npy':
        normal = np.load(path)
    else:
        raise OSError(f'Unknown normal file type "{file_ext}"')
    return normal

def resize_images(images, dsize, interpolation=cv2.INTER_CUBIC):
    resized = []
    for i in range(images.shape[0]):
        image = images[i]
        image = cv2.resize(image, dsize=dsize, interpolation=interpolation)
        resized.append(image)
    resized = np.stack(resized, 0)
    return resized

def normalize_augment(polar, mask):
    if len(mask.shape) == 3:
        mask = mask[...,0]
    n, h, w = polar.shape[:3]
    polar = np.reshape(polar, (n, h * w, -1))

    temp = np.mean(polar[:, mask.flatten()==1,:], axis=2, keepdims=True)
    temp = np.max(temp, axis=(0,1), keepdims=True)
    polar /= (temp + 1.0e-6)
    polar = polar.reshape(n, h, w, -1)
    return polar

def crop_augment(mask, images, margin=8, patch_size=256, max_res=2048, mask_flag=True):
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    h0, w0 = mask.shape[:2]
    
    if mask_flag:
        rows, cols = np.nonzero(mask)
        rowmin, rowmax = np.min(rows), np.max(rows)
        rownum = rowmax - rowmin
        colmin, colmax = np.min(cols), np.max(cols)
        colnum = colmax - colmin
        if rowmin-margin<=0 or rowmax+margin>h0 or colmin-margin<=0 or colmax+margin>w0:
            crop_flag = False
        else:
            crop_flag = True

        if crop_flag:
            if rownum > colnum:
                r_s = rowmin - margin
                r_e = rowmax + margin
                c_s = np.max([colmin-int(0.5*(rownum-colnum))-margin, 0])
                c_e = np.min([colmax+int(0.5*(rownum-colnum))+margin, w0])
            elif colnum >= rownum:
                r_s = np.max([rowmin-int(0.5*(colnum-rownum))-margin, 0])
                r_e = np.min([rowmax+int(0.5*(colnum-rownum))+margin, h0])
                c_s = colmin - margin
                c_e = colmax + margin  
        else:
            r_s, r_e = 0, h0
            c_s, c_e = 0, w0
    else:
        margin = 0
        rowmin, rowmax, rownum = 0, h0, h0
        colmin, colmax, colnum = 0, w0, w0
        if rownum > colnum:
            r_s = int(0.5 * rownum) - int(0.5 * colnum)
            r_e = int(0.5 * rownum) + int(0.5 * colnum)
            c_s, c_e = colmin, colmax
        elif colnum >= rownum:
            r_s = rowmin-margin
            r_e = rowmax+margin
            c_s = int(0.5 * colnum) - int(0.5 * rownum)
            c_e = int(0.5 * colnum) + int(0.5 * rownum)  
               
    mask = mask[r_s:r_e, c_s:c_e]
    images_cropped = images[:, r_s:r_e, c_s:c_e]
    h_crop, w_crop = mask.shape[:2]
    h = int(np.floor(np.max([h_crop, w_crop]) / patch_size) * patch_size)
    h = max(min(h, max_res), patch_size)
    w = h
    mask = np.float32(cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_CUBIC)> 0.5)
    roi = np.array([h0, w0, r_s, r_e, c_s, c_e])
    images_resized = resize_images(images_cropped, (w, h), cv2.INTER_CUBIC)
    return mask, images_resized, roi