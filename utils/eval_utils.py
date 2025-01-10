import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F

import utils.utils as util

pi = 3.141592653

def show_est(args, 
             data, 
             num_iter, 
             res_dict, 
             res_path_dict,
             save_indv=False, 
             save_overview=False):
    # the sequence of show_list influence the order of overview image
    show_name_list = [
        'normal_pred',
    ]
    required_mask_set = {
        'polar', 'normal_pred'
    }
    multi_color_image_set = {
        'polar', 
    }
    normal_set = {'normal_pred'}
    kw2name = {
        'polar': 'polar', 
        'normal_pred': f'{args.normal_model}_normal',
        'material_est_pred': f'{args.normal_model}_material',
    }

    show_list = [k for k in show_name_list if k in data]
    if num_iter == 0: args.log.wprint(show_list)
    if num_iter >= args.save_max_num:
        save_indv, save_overview = False, False
    bs, _, h, w = data['mask'].shape
    ddev = data['mask'].device
    save_c = 3
    save_num = min(bs, 16)

    # save the concat img
    if 'names' not in res_path_dict:
        res_path_dict['names'] = []
    for i in range(save_num):
        res_path_dict['names'].append(data['name'][i])
    if save_overview or save_indv:

        sep_h = torch.ones((save_c, 3, w), device=ddev)
        sep_w = torch.ones((save_c, (h+3)*save_num, 3), device=ddev)
        res_stack = []
        for keyword in show_list:
            if keyword not in res_path_dict:
                res_path_dict[keyword] = []

            if keyword in multi_color_image_set:
                if keyword in required_mask_set:
                    data[keyword][:save_num] *= data['mask'][:save_num, None]
                if save_overview:
                    tensor_stack = data[keyword][:save_num, 0].expand(-1, save_c, -1, -1)
                    tensor_stack = torch.cat([torch.cat([tensor_stack[i], sep_h], 1) for i in range(save_num)], 1)
                if save_indv:
                    npy_stack = (data[keyword][:save_num].flip(2).clamp(0, 1)**(1/2.2)*255).permute((0, 1, 3, 4, 2)).cpu().data.numpy().astype(np.uint8)
                    for i in range(save_num):
                        num = npy_stack.shape[1]
                        for k in range(num):
                            img = npy_stack[i, k]
                            save_path = os.path.join(args.log_dir, f"{data['name'][i]}_{kw2name[keyword]}{k+1}.png")
                            cv2.imwrite(save_path, img)
                        res_path_dict[keyword].append(os.path.basename(save_path))

            elif keyword in normal_set:
                normal_tensor = (data[keyword][:save_num] + 1.)/2. * data['mask'][:save_num]
                if save_overview:
                    tensor_stack = torch.cat([torch.cat([normal_tensor[i], sep_h], 1) for i in range(save_num)], 1)
                    tensor_stack = tensor_stack ** 2.2
                if save_indv:
                    npy_stack = data[keyword][:save_num].permute((0, 2, 3, 1)).cpu().data.numpy()
                    mask = data['mask'][:save_num].permute((0, 2, 3, 1)).cpu().data.numpy()
                    for i in range(save_num):
                        save_path = os.path.join(args.log_dir, f"{data['name'][i]}_{kw2name[keyword]}.png")
                        util.save_normal(npy_stack[i], mask[i], save_path)
                        res_path_dict[keyword].append(os.path.basename(save_path))  
            
            else: 
                continue

            if save_overview:
                res_stack.append(tensor_stack)
                res_stack.append(sep_w)

        if save_overview and res_stack:
            del res_stack[-1]
            res_img = (torch.cat(res_stack, 2).flip(0).clamp(0, 1)**(1/2.2)*255).permute((1, 2, 0)).cpu().data.numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(args.log_dir, f'iter{num_iter}_overview.jpg'), res_img)
    