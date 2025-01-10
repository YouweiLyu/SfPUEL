import datetime, time, os
from . import utils

class Logger(object):
    def __init__(self, args):
        self.args_abbv = {
            'epochs':'ep', 'lr_init':'lr', 'lr_decay':'lr_dec'}
        self.times = {'init': time.time()}
        self.local_rank = args.local_rank
        if args.make_dir:
            self._setup_dirs(args)
        self.args = args
        args.log  = self
        self.print_args()

    def print_args(self):
        strs = ''.join([
            '------------ Options -------------\n',
            utils.dict2string(vars(self.args)),
            '-------------- End ----------------\n',
        ])
        self.write(strs, True)

    def _add_arguments(self, args):
        arg_var  = vars(args)
        info_list = []
        for k in args.str_keys:
            if isinstance(arg_var[k], str):
                info_list.append(f',{arg_var[k]}')
            elif isinstance(arg_var[k], list):
                info_list.append(',in-' if 'in_list' in k else '')
                info_list.append(
                    ''.join(
                        ''.join(s.split('_')).capitalize() for s in arg_var[k]))
        for k in args.bool_keys: 
            info_list.append(f',{k}' if arg_var[k] else '')
        for k in args.val_keys:
            var_key = self.args_abbv.get(k, f'{k[:2]}_{k[-1]}')
            info_list.append(f',{var_key}-{arg_var[k]}')
        info = ''.join(info_list)
        return info

    def _setup_dirs(self, args):
        date_now = datetime.datetime.now()
        dir_name = f'{args.suffix}' if args.suffix else dir_name
        dir_name += self._add_arguments(args)
        dir_name += ',DEBUG' if args.debug else ''

        self._check_path(args, dir_name)
        file_dir = os.path.join(
            args.log_dir, 
            '{},{},dev{}.log'.format(
                dir_name, date_now.strftime('%H;%M;%S'), self.local_rank)
        )
        self.log_fie = open(file_dir, 'w')
        return 

    def _check_path(self, args, dir_name):
        if args.resume and os.path.isfile(args.resume):
            log_root = os.path.join(os.path.dirname(args.resume), dir_name)
        else:
            if args.debug:
                dir_name = 'DEBUG/' + dir_name
            log_root = os.path.join(args.save_root, args.item, dir_name)
        if args.split == 'train':
            args.log_dir = os.path.join(log_root)
            args.ckpt_dir = os.path.join(log_root, 'ckpt')
            utils.make_folders([args.log_dir, args.ckpt_dir])
        elif args.split == 'test':
            args.log_dir, args.ckpt_dir = log_root, log_root
            utils.make_folders([args.log_dir])

    def wprint(self, strs, main_print=False):
        """output the string to logger and Stdout
        """
        if not (main_print and self.local_rank):
            print(str(strs))
            if self.args.make_dir:
                self.log_fie.write(f'{strs}\n')
                self.log_fie.flush()

    def write(self, strs, main_print=False):
        if not (main_print and self.local_rank):
            if self.args.make_dir:
                self.log_fie.write(f'{strs}\n')
                self.log_fie.flush()
