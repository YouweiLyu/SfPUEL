import argparse
import numpy as np

class BaseOpts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        #### Trainining Dataset ####
        self.parser.add_argument('--split', default='train', type=str)
        self.parser.add_argument('--real_dataloader', default='Real_Dataloader')

        # parameters about the datasets
        self.parser.add_argument('--data_dir_trn', default='')
        self.parser.add_argument('--data_dir_val', default='')
        self.parser.add_argument('--data_dir_tst', default='')
        self.parser.add_argument('--randcrop', default=0, type=int)
        self.parser.add_argument('--grayscale', default=False, action='store_true')
        self.parser.add_argument('--workers', default=1, type=int)
        #### Device Arguments ####
        self.parser.add_argument('--multi_gpu', default=False, action='store_true')
        self.parser.add_argument('--num_gpus', default=2, type=int)
        self.parser.add_argument('--time_sync', default=False, action='store_true')
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--cuda', default=True, type=bool)
        #### Global ctl params ####
        self.parser.add_argument('--in_normal_dir', default='normal', type=str)
        #### Normal Model Arguments ####
        self.parser.add_argument('--normal_model', default='SfPUEL')
        self.parser.add_argument('--normal_resume', default=None)
        self.parser.add_argument('--normal_pretrain', default='data/checkpoints/ckpt.pth')
        self.parser.add_argument('--normal_dataloader', default='normal_dataloader')
        self.parser.add_argument('--normal_inputs', default=['polar','mask'], nargs='+', type=str)
        self.parser.add_argument('--normal_inputs_2', default=['stokes'], nargs='+', type=str)
        self.parser.add_argument('--normal_outputs', default=['normal','material_label'], nargs='+', type=str)
        # resume parameters
        self.parser.add_argument('--resume', default=None)
        #### Displaying & Saving Arguments ####
        self.parser.add_argument('--disable_rectify_outputs', default=True, action='store_false')
        self.parser.add_argument('--save_intv', default=1, type=int)
        self.parser.add_argument('--train_disp', default=50, type=int)
        self.parser.add_argument('--train_save', default=50, type=int)
        self.parser.add_argument('--save_start_epoch', default=0, type=int)
        self.parser.add_argument('--val_intv', default=1, type=int)
        self.parser.add_argument('--val_disp', default=50, type=int)
        self.parser.add_argument('--val_save', default=50, type=int)
        self.parser.add_argument('--max_train_iter', default=-1, type=int)
        self.parser.add_argument('--max_val_iter', default=-1, type=int)
        self.parser.add_argument('--max_test_iter', default=-1, type=int)
        self.parser.add_argument('--train_save_n', default=5, type=int)
        self.parser.add_argument('--test_save_n', default=5, type=int)
        #### Log Arguments ####
        self.parser.add_argument('--save_root', default='results')
        self.parser.add_argument('--item', default='')
        self.parser.add_argument('--suffix', default='SfPUEL')
        self.parser.add_argument('--debug', default=False, action='store_true')
        self.parser.add_argument('--make_dir', default=True)
        self.parser.add_argument('--save_split', default=False, action='store_true')
        # UniPS-SDM network Configs
        self.parser.add_argument('--pixel_samples', type=int, default=2048)
        self.parser.add_argument('--decoder_resolution', type=int, default=512)
        self.parser.add_argument('--canonical_resolution', type=int, default=256)
        # self.parser.add_argument('--scalable', default=False, action='store_true')

    def set_default(self):
        if self.args.debug:
            self.args.train_disp = 1
            self.args.train_save = 1
            self.args.val_save = 1
            self.args.val_disp = 1
            self.args.max_train_iter = 3
            self.args.max_val_iter = 3
            self.args.max_test_iter = 3
            self.args.test_intv = 1
            self.args.save_intv = np.inf
        
        self.args.real_inputs = self.args.normal_inputs
        self.args.model_outputs = set(self.args.normal_outputs)
        if self.args.normal_inputs_2:
            self.args.normal_branch_inputs = [self.args.normal_inputs, self.args.normal_inputs_2]
        else:
            self.args.normal_branch_inputs = [self.args.normal_inputs]
        
        self.args.max_train_iter = np.inf if self.args.max_train_iter==-1 else self.args.max_train_iter
        self.args.max_test_iter = np.inf if self.args.max_test_iter==-1 else self.args.max_test_iter
        self.args.max_val_iter = np.inf if self.args.max_val_iter==-1 else self.args.max_val_iter

    def collect_info(self):
        self.args.str_keys  = []
        self.args.val_keys  = []
        self.args.bool_keys = []

    def parse_loaded_data_names(self):
        all_names = self.args.normal_inputs + self.args.normal_inputs_2
        if self.args.split == 'train':
            all_names += self.args.model_outputs
        elif self.args.split == 'test':
            all_names += self.args.test_loaded_data
        else:
            raise Exception()
        load_names = ['name']
        for item in all_names:
            if item in ['normal', 'material_label']:
                load_names.append(f'{item}_gt')
            elif 'polar' in item:
                load_names.append('polar')
            elif 'stokes' in item:
                load_names.append('polar')
            elif item in ['mask']:
                load_names.append(item)
            else:
                raise ValueError(f'{item} couldnot be converted to loaded name')
        self.args.loaded_data_names = set(load_names)

    def parse(self):
        self.args = self.parser.parse_args()
        self.set_default()
        self.parse_loaded_data_names()
        self.collect_info()

class TestOpts(BaseOpts):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()
        self.parser.add_argument('--test_batch', default=1, type=int)
        self.parser.add_argument('--test_loaded_data', default=['normal'], nargs='+', type=str)
        self.parser.add_argument('--save_res_dir', default=None)
        self.parser.add_argument('--save_indv', default=True, action='store_true')
        self.parser.add_argument('--test_shuffle', default=False, action='store_true')
        self.parser.add_argument('--save_overview',  default=False, action='store_true')
        self.parser.add_argument('--save_max_num', default=-1, type=int)

    def set_default(self):
        super().set_default()
        self.args.test_save_n = min(self.args.test_batch, self.args.test_save_n)
        self.args.split = 'test'
        self.args.save_max_num = np.uint64(self.args.save_max_num)
        self.args.save_root = 'results'
        self.args.drop_rate = 0

    def collect_info(self): 
        self.args.str_keys  = []
        self.args.val_keys  = []
        self.args.bool_keys = []

    def parse(self):
        super().parse()
        return self.args