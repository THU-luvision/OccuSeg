from optparse import OptionParser
def get_args():
    parser = OptionParser()
    parser.add_option('--batch_size', dest='batch_size', default=4, type='int', help='batch size')
    parser.add_option('--optim', dest='optim', default='Adam', help='optimizer type')
    parser.add_option('--loss', dest='loss', default='cross_entropy', help='loss type')
    parser.add_option('--lr', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('--regress_sigma', dest='regress_sigma', default=0.2, type='float', help='distance to center variance')
    parser.add_option('--regress_weight', dest='regress_weight', default=100, type='float', help='distance to center variance')
    parser.add_option('--displacement_weight', dest='displacement_weight', default=100, type='float', help='distance to center variance')
    parser.add_option('--backbone_network', dest='backbone_network', default='LearnBWNet.pth', help='backbone network structure')

    parser.add_option('--checkpoint_file', dest='load', default=False, help='load file model')
    parser.add_option('--checkpoint', dest='checkpoint', default=0, type='int', help='snapshot')
    parser.add_option('--dataset', dest='dataset', default='scannet', help='dataset type')
    parser.add_option('--gamma', dest='gamma', type='float', default=0, help='lr decay')
    parser.add_option('--step_size', dest='step_size', type='int', default=60000, help='step_size')
    parser.add_option('--max_epoch', default=2000, type='int', help='max_epoch')
    parser.add_option('--checkpoints_dir', dest='checkpoints_dir', default='./ckpts/', help='checkpoints_dir')
    parser.add_option('--snapshot', dest='snapshot', default=1, type='float', help='snapshot every x epoch')
    parser.add_option('--display', dest='display', default=10, type='float', help='display')

    # tensorboardX visualization
    # parser.add_option('--tensorboard_log_dir', type='str', default='./log')
    parser.add_option('--taskname', type='str', default='default_name')

    # Sparse Conv Hyper-parameters
    # m = 16  # 16 or 32
    # residual_blocks = False  # True or False
    # block_reps = 1  # Conv block repetition factor: 1 or 2
    # eval_epoch = 10
    # eval_save_ply = True  # save ply file when evaluating training
    # training_epochs = 1024

    # scale = 20,  # Voxel size = 1/scale
    # val_reps = 1,  # Number of test views, 1 or more
    # batch_size = 4,
    # dimension = 3,
    # full_scale = 4096,

    parser.add_option('--m', default=16, type='int', help='16 or 32')
    parser.add_option('--residual_blocks', default=False, action='store_true')
    parser.add_option('--block_reps', default=1, type='int', help='Conv block repetition factor: 1 or 2')
    parser.add_option('--kernel_size', default=3, type='int', help='Kernel Size')

    parser.add_option('--scale', default=20, type='int', help='Voxel size = 1/scale')
    parser.add_option('--val_reps', default=1, type='int', help='Number of test views, 1 or more')
    parser.add_option('--dimension', default=3, type='int', help='only 3 is supported')
    parser.add_option('--full_scale', default=4096, type='int')
    parser.add_option('--rotation_guide_level', default=0, type='int', help='Kernel Size')


    parser.add_option('--evaluate', default=False, action='store_true')
    parser.add_option('--use_dense_model', default=False, action='store_true')

    parser.add_option('--use_rotation_noise', default=False, action='store_true')
    parser.add_option('--use_elastic', default=False, action='store_true')
    parser.add_option('--use_normal', default=False, action='store_true')
    parser.add_option('--use_full_normal', default=False, action='store_true')
    parser.add_option('--simple_train', default=False, action='store_true')

    parser.add_option('--all_to_train', default=False, action='store_true')

    # test use only
    parser.add_option('--test_path',default='', type='str')
    parser.add_option('--test_result_path',default='', type='str')

    #input feature
    parser.add_option('--use_feature',default='c', type='str',help='c:color, d:depth, n:normal as feature')

    (options, args) = parser.parse_args()
    print(args)
    return options


def ArgsToConfig(args):
    config = {}
    config['m'] = args.m
    config['use_full_normal'] = args.use_full_normal
    config['residual_blocks'] = args.residual_blocks
    config['simple_train'] = args.simple_train
    config['block_reps'] = args.block_reps
    config['batch_size'] = args.batch_size

    config['scale'] = args.scale
    config['val_reps'] = args.val_reps
    config['dimension'] = args.dimension
    config['full_scale'] = args.full_scale
#    config['unet_structure'] = [args.m, 2 * args.m, 3 * args.m, 4 * args.m, 6 * args.m, 8 * args.m, 12 * args.m]
#    config['unet_structure'] = [2 * args.m, 2.5 * args.m, 3 * args.m, 4 * args.m, 5 * args.m, 6 * args.m, 7 * args.m]

    config['unet_structure'] = [args.m, 2 * args.m, 3 * args.m, 4 * args.m, 5 * args.m, 6 * args.m]
    config['kernel_size'] = args.kernel_size
    config['use_rotation_noise'] = args.use_rotation_noise
    config['checkpoint'] = args.checkpoint
    config['checkpoints_dir'] = args.checkpoints_dir
    config['max_epoch'] = args.max_epoch
    config['snapshot'] = args.snapshot
    config['optim'] = args.optim
    config['loss'] = args.loss
    config['lr'] = args.lr
    config['gamma'] = args.gamma
    config['step_size'] = args.step_size
    config['rotation_guide_level'] = args.rotation_guide_level
    config['evaluate'] = args.evaluate
    config['backbone_network'] = args.backbone_network

    config['use_normal'] = args.use_normal
    config['use_elastic'] = args.use_elastic
    config['use_feature'] = args.use_feature
    config['use_dense_model'] = args.use_dense_model
    config['regress_sigma'] = args.regress_sigma
    config['regress_weight'] = args.regress_weight
    config['displacement_weight'] = args.displacement_weight
    # c: color, n: normal, d: depth define in tangentnet, h: height(z axis value)
    config['input_feature_number']=0
    if 'l' in config['use_feature']:
        config['input_feature_number']+=3
    if 'c' in config['use_feature']:
        config['input_feature_number']+=3
    if 'n' in config['use_feature']:
        config['input_feature_number']+=3
    if 'd' in config['use_feature']:
        config['input_feature_number']+=9
    if 'h' in config['use_feature']:
        config['input_feature_number']+=1

    config['dataset'] = args.dataset

    return config
