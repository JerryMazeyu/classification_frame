class Config(object):
    data_path = './data/eminist/'
    train_data_path = './data/eminist/train/'
    test_data_path = './data/eminist/test/'
    num_workers = 2
    num_class = 47
    batch_size = 5
    max_epoch = 1
    lr = 0.002
    weight_decay = 1e-4
    use_gpu = False
    print_freq = 20
    vis = True
    env = 'minist'
    net_path = './checkpoints/densenet121_finetune_wt_1.pth'
    id2class = None

opt = Config()