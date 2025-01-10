import os 
import numpy as np
import glob
import pandas as pd
import tqdm
import argparse

import utils.utils as util
import utils.file_utils as futil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', default='SfPUEL')
    parser.add_argument('--test_data_dir', '-d', required=True)
    parser.add_argument('--gt_data_dir', required=True)
    args = parser.parse_args()
    
    test_data_name = os.path.basename(args.test_data_dir)
    table_path = os.path.join(os.path.dirname(args.test_data_dir), f'{test_data_name}_normal_eval.xlsx')
    normal_paths = sorted(glob.glob(f'{args.test_data_dir}/*{args.method}_normal.png'))
    if not normal_paths:
        raise FileNotFoundError(args.test_data_dir)
    
    columns=['Name', 'MAngE', 'MedianAngErr', 'ARMSE', '11.25°Acc', '22.5°Acc', '30°Acc']
    filenames = []
    mean_angle_err = []
    median_angle_err = []
    rmse_angle_err = []
    acc_11_25 = []
    acc_22_50 = []
    acc_30_00 = []

    for normal_pred_path in tqdm.tqdm(normal_paths):
        filename = futil.get_filename(normal_pred_path, args.method)
        normal_gt_path = futil.get_normal_gt_path(filename, args.gt_data_dir)
        mask_path = futil.get_mask_path(filename, args.gt_data_dir)
        normal_pred = futil.read_normal(normal_pred_path)
        normal_gt = futil.read_normal(normal_gt_path)
        normal_pred = util.unify_shape(normal_gt, normal_pred, 'normal')
        mask = futil.read_mask(mask_path)
        
        errmap = np.rad2deg(np.arccos((normal_pred * normal_gt).sum(axis=-1).clip(-1,1)))
        angle_err = (errmap[mask]).flatten()
        mean_angle_err.append(np.mean(angle_err))
        median_angle_err.append(np.median(angle_err))
        rmse_angle_err.append(np.sqrt((angle_err ** 2).mean()))
        acc_11_25.append((angle_err < 11.25).mean())
        acc_22_50.append((angle_err < 22.5).mean())
        acc_30_00.append((angle_err < 30).mean()) 
        filenames.append(filename)

    table = [filenames, mean_angle_err, median_angle_err, rmse_angle_err, acc_11_25, acc_22_50, acc_30_00]
    df = pd.DataFrame(table).transpose()
    df.columns = columns
    df.to_excel(table_path, columns=columns, index=False)        
