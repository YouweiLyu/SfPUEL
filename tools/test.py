import torch

from options.base_opts import TestOpts as Opts
from utils.logger import Logger
from datasets import Dataloader
from models   import model_builder
from models import model_utils as mutil
from utils import eval_utils as eutil

def main(args):
    args.local_rank = 0
    local_device = f'cuda:{args.local_rank:d}' if args.cuda else 'cpu'
    args.device = torch.device(local_device)
    
    Logger(args)
    model = model_builder.build_normal_model(args)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    test_loader = Dataloader.loader_builder(args, args.normal_dataloader, args.split)
    num_iter = min(test_loader.__len__(), args.max_test_iter)
    args.log.wprint('=>\tTest Iter = {}'.format(num_iter))

    res_path_dict = {}
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            data = mutil.load_data(sample, args.device)
            inputs = mutil.get_inputs(data, args.normal_branch_inputs)
            pred = model(inputs, args.decoder_resolution, args.canonical_resolution, args.pixel_samples)
            mutil.update_data(data, pred, args.split)
            mutil.postprocess_data(data, args.split)
            eutil.show_est(
                args, 
                data, 
                idx, 
                None, 
                res_path_dict,
                args.save_indv, 
                args.save_overview,
            )
            if idx + 1 >= num_iter: break


if __name__ == '__main__':
    args = Opts().parse()
    torch.manual_seed(args.seed)
    main(args)
