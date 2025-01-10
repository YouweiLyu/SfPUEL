from torch.utils.data import DataLoader

def loader_builder(args, dataset_name, split='train'):
    collate_func = None
    dataset_split = ''

    datasets = __import__('datasets.' + dataset_name)
    dataset_file = getattr(datasets, dataset_name)
    args_tst = [args, args.data_dir_tst, 'test']

    if split == 'test':
        test_set = getattr(dataset_file, dataset_name+dataset_split)(*args_tst)
        test_loader  = DataLoader(test_set, batch_size=args.test_batch,
            num_workers=args.workers, pin_memory=args.cuda, shuffle=args.test_shuffle,
            collate_fn=collate_func,)
        args.log.wprint('=> fetching test data in {}'.format(args.data_dir_tst),True)
        args.log.wprint('Found Data:\t {} Test & Test Batch: {} '.format(len(test_set), args.test_batch),True)
        args.log.wprint('\t Test Batch: {}'.format(args.test_batch),True)
        return test_loader
    else:
        raise Exception('=> Unknown split type for Loading {}'.format(split))
