import os
import string
import random
import logging
import torch
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import methods
import datasets

# parse parameters

def parse_args(parser):
    args = parser.parse_args()
    args.method_params = parse_params(args.method_params)
    args.train_dataset_params = parse_params(args.train_dataset_params)
    args.eval_dataset_params = parse_params(args.eval_dataset_params)
    args.optimizer_params = parse_params(args.optimizer_params)
    args.scheduler_params = parse_params(args.scheduler_params)
    return args

def parse_params(params):
    # parse key-value parameters
    return dict(zip(params[::2],
                    [convert_value(value) \
                     for value in params[1::2]]))

def convert_value(value):
    # convert to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # convert to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # convert to bool
    if value.lower() in ['true', 'yes']:
        return True
    elif value.lower() in ['false', 'no']:
        return False
    
    # convert to list
    value = value.split(',')
    if len(value) > 1:
        for i in range(len(value)):
            value[i] = convert_value(value[i])
    else:
        value = value[0]
    
    # if cannot convert
    return value

# get logger

def create_exp_dir(args):
    exp_dir = args.log_dir
    
    for next_level_dir in ['', args.train_dataset]:
        exp_dir = os.path.join(exp_dir, next_level_dir)
        build_dirs(exp_dir)
    if 'noise_type' in args.train_dataset_params.keys():
        exp_dir = os.path.join(exp_dir, args.train_dataset_params['noise_type'])
        build_dirs(exp_dir)
    if 'noise_rate' in args.train_dataset_params.keys():
        exp_dir = os.path.join(exp_dir, f'{args.train_dataset_params["noise_rate"]}')
        build_dirs(exp_dir)
    exp_dir = os.path.join(exp_dir, args.method)
    build_dirs(exp_dir)
    
    return exp_dir

def create_exp_name(args):
    exp_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id = generate_experiment_id()
    exp_name = '{}__{}'.format(exp_time, exp_id)
    ignore_list = ['num_str_aug_view',
                   'dim_ins', 
                   'dim_ins_hidden',
                   'num_ins_layer',
                   'dim_sem',
                   'dim_sem_hidden',
                   'num_sem_layer']
    for key, value in args.method_params.items():
        if key in ignore_list:
            continue
        exp_name += '__{}_{}'.format(key, value)
    exp_name += '__seed_{}'.format(args.seed)
    return exp_name

def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def generate_experiment_id(length=4):
    exp_id = ''
    for _ in range(length):
        choosen_str = random.SystemRandom().choice(string.ascii_uppercase + \
                                                   string.digits)
        exp_id += choosen_str
    return exp_id

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# get tensorboard writer

def setup_writer(args, exp_name):
    if not args.use_tensorboard:
        return None
    
    writer_path = os.path.join(args.tensorboard_dir, args.train_dataset)
    if 'noise_type' in args.train_dataset_params.keys():
        writer_path = os.path.join(writer_path, args.train_dataset_params['noise_type'])
    if 'noise_rate' in args.train_dataset_params.keys():
        writer_path = os.path.join(writer_path, f'n{args.train_dataset_params["noise_rate"]}')
    writer_path = os.path.join(writer_path, args.method)
    writer_path = os.path.join(writer_path, exp_name)

    writer = SummaryWriter(writer_path)
    return writer

# log basic experiment config
def log_experiment_config(args, logger):
    logger.info('[Basic Config]')

    args_dict = vars(args)

    for arg_name, arg_value in args_dict.items():
        if isinstance(arg_value, dict):
            logger.info(f'{arg_name}:')
            for key, value in arg_value.items():
                logger.info(f'    {key}: {value}')
        else:
            logger.info(f'{arg_name}: {arg_value}')

# get method
def get_method(args):
    name = args.method
    num_classes = args.num_classes
    method_params = args.method_params

    if hasattr(methods, name):
        method = getattr(methods, name)(num_classes=num_classes, **method_params)
    else:
        raise RuntimeError(f'Wrong method name: {name}')
    
    return method

# get dataloader

def get_dataloader_train(args):
    name = args.train_dataset
    params = args.train_dataset_params

    if hasattr(datasets, name):
        dataset = getattr(datasets, name)(**params)
    else:
        raise RuntimeError(f'Wrong trian_dataset name: {name}')
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=params['batch_size'],
                            shuffle=True,
                            num_workers=params['num_workers'])
    return dataloader

def get_dataloader_eval(args):

    name = args.eval_dataset
    params = args.eval_dataset_params
    
    if hasattr(datasets, name):
        dataset = getattr(datasets, name)(**params)
    else:
        raise RuntimeError(f'Wrong eval_dataset name: {name}')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=params['batch_size'],
                            shuffle=False,
                            num_workers=params['num_workers'])
    return dataloader

# get model

def get_model(args):
    name = args.model
    num_classes = args.num_classes

    if name == 'resnet18':
        from models import resnet18
        model = resnet18(num_classes)
    elif name == 'resnet34':
        from models import resnet34
        model = resnet34(num_classes)
    else:
        raise RuntimeError(f'Wrong model name: {name}')
    
    return model

# get optimizer

def get_optimizer(args, model):
    name = args.optimizer
    params = args.optimizer_params
    if name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=params['lr'],
                              momentum=params['momentum'],
                              weight_decay=params['weight_decay'],
                              nesterov=params['nesterov'])
    else:
        raise RuntimeError(f'Wrong optimizer name: {name}')
    
    return optimizer

# get scheduler

def get_scheduler(args, optimizer):
    name = args.scheduler
    params = args.scheduler_params
    if name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=params['T_max'],
                                                         eta_min=params['eta_min'])
    elif name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=params['step_size'],
                                              gamma=params['gamma'])
    elif name == 'multisteplr':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=params['milestones'],
                                                   gamma=params['gamma'])
    else:
        raise RuntimeError(f'Wrong scheduler name: {name}')

    return scheduler
