import os
import torch
import argparse
import random
import utils
import numpy as np
import time
from time import sleep
from trainer import Trainer
from evaluator import Evaluator

parser = argparse.ArgumentParser()
# method
parser.add_argument('--method', type=str, default='ce')
parser.add_argument('--method_params', type=str, nargs='+', action='extend',
                    metavar=('KEY', 'VALUE'), help='parameters for method')
# model
parser.add_argument('--model', type=str, default='resnet18')
# datasets
parser.add_argument('--train_dataset', type=str, default='cifar10')
parser.add_argument('--train_dataset_params', type=str, nargs='+', action='extend',
                    metavar=('KEY', 'VALUE'), help='parameters for train_dataset')
parser.add_argument('--eval_dataset', type=str, default='cifar10')
parser.add_argument('--eval_dataset_params', type=str, nargs='+', action='extend',
                    metavar=('KEY', 'VALUE'), help='parameters for eval_dataset')
parser.add_argument('--num_classes', type=int, default=10)
# optimize
parser.add_argument('--total_epoch', type=int, default=200)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--optimizer_params', type=str, nargs='+', action='extend',
                    metavar=('KEY', 'VALUE'), help='parameters for optimizer')
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--scheduler_params', type=str, nargs='+', action='extend',
                    metavar=('KEY', 'VALUE'), help='parameters for scheduler')
# others
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', action='extend', nargs='+', type=str, required=True)
parser.add_argument('--use_tensorboard', action='store_true', default=False)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--write_freq', type=int, default=100)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--tensorboard_dir', type=str, default='runs')
parser.add_argument('--save_model', action='store_true', default=False)

# parse args
args = utils.parse_args(parser)

# setup logger
exp_dir = utils.create_exp_dir(args)
exp_name = utils.create_exp_name(args)
exp_log_file = os.path.join(exp_dir, f'{exp_name}.log')
exp_model_weight_file = os.path.join(exp_dir, f'{exp_name}_weight.pth')
logger = utils.setup_logger(name=exp_name,
                            log_file=exp_log_file)

# setup tensorboard writer
writer = utils.setup_writer(args, exp_name)

# setup random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# setup device
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0')

# log basic experiment config
utils.log_experiment_config(args, logger)

def main():
    # setup model
    model = utils.get_model(args)
    model = model.to(device)

    # setup dataset
    train_dataloader = utils.get_dataloader_train(args)
    eval_dataloader = utils.get_dataloader_eval(args)

    # setup method
    method = utils.get_method(args)

    # setup optimizer & scheduler
    optimizer = utils.get_optimizer(args, model)
    scheduler = utils.get_scheduler(args, optimizer)
    
    # setup trainer
    trainer = Trainer(train_dataloader, logger, writer, device, args)
    
    # setup evaluator
    evaluator = Evaluator(eval_dataloader, logger, writer, device)
    
    # log dataset sample number
    logger.info('Sample Num of Train Dataset:')
    logger.info(f'    {args.train_dataset}: {len(train_dataloader.dataset)}')
    logger.info('Sample Num of Eval Dataset:')
    logger.info(f'    {args.eval_dataset}: {len(eval_dataloader.dataset)}')
    logger.info('')
    
    # start training
    time_train = 0
    time_eval = 0
    logger.info('[Training]')
    for epoch in range(args.total_epoch):
        # train
        time_train_start = time.time()
        logger.info('+' + 'Train'.center(30, '-') + '+')
        trainer.train(model, optimizer, method, epoch + 1)
        scheduler.step()
        time_train += time.time() - time_train_start
        
        # eval
        time_eval_start = time.time()
        if (epoch + 1) % args.eval_freq == 0 \
            or epoch + 1 == 1 \
            or epoch + 1 == args.total_epoch \
            or args.total_epoch - (epoch + 1) < 10:
            logger.info('+' + 'Eval'.center(30, '-') + '+')
            evaluator.eval(model, epoch + 1)
        time_eval += time.time() - time_eval_start
    
    avg_last_10_eval_acc = sum(evaluator.last_10_acc.items) / len(evaluator.last_10_acc.items)

    logger.info('+' + 'Finish training process'.center(30, '-') + '+')
    logger.info(f'Avg of the last 10 Eval Acc: {avg_last_10_eval_acc}')
    logger.info(f'Training time: {time_train // 60} mins')
    logger.info(f'Eval time: {time_eval // 60} mins')
    logger.info('')
    
    logger.info('[Extra]')
    if writer is not None:
        # close writer
        logger.info('+' + 'Close Writer'.center(30, '-') + '+')
        writer.close()
    
    if args.save_model:
        logger.info('+' + 'Saving model'.center(30, '-') + '+')
        # save model weight
        torch.save(model.state_dict(), exp_model_weight_file)
    logger.info('')
    
    logger.info('[All process finish]')
    sleep(3)

if __name__ == '__main__':
    main()