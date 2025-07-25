import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    ''' # For Resnet
    cfg_dict = {
        'optim': 'Adam',
        'schedule': 'StepLR',
        'lr': 2e-5,
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '15',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    '''
    # cfg_dict = {
    #     'optim': 'SGD', # 'Adam'
    #     'schedule': 'StepLR',
    #     'lr': 0.05,  # '1e-5'
    #     'batch': 8,
    #     'ave_batch': 1,
    #     'epoch': 30,
    #     'step_size': '40',
    #     'gamma': 0.5,
    #     'clip_gradient': 0,
    #     'test_batch': 1
    # }

    cfg_dict = {
        'optim': 'SGD', # 'Adam'
        'schedule': 'StepLR',
        'lr': 0.005,  # '1e-5'
        'batch': 8,
        'ave_batch': 1,
        'epoch': 25,
        'step_size': '15,20',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1
    }
    


    parser = base_config(cfg_dict)
    # Add custom params here
    # parser.add_argument('--size', default=320, type=int, help='Input size')
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    # Config post-process
    #config['params'] = [['encoder', config['lr'] / 10], ['global_', config['lr']], ['region', config['lr']], ['local', config['lr']]]
    config['params'] = [['encoder', config['lr'] / 10], ['decoder', config['lr']]]
    config['lr_decay'] = 0.9
    
    return config, None