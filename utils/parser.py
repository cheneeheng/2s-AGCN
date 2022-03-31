
import argparse as ap
from utils.utils import str2bool

__all__ = ['get_parser']


def get_parser() -> ap.ArgumentParser:

    # parameter priority: command line > config > default
    p = ap.ArgumentParser(description='Skeleton-based action recognition')

    p.add_argument('--config',
                   default='./config/nturgbd-cross-view/test_bone.yaml',
                   help='path to the configuration file')

    # setup
    p.add_argument('--work-dir',
                   default='./work_dir/temp',
                   help='the work folder for storing results')
    p.add_argument('--model-saved-name',
                   default='',
                   help='the folder to store events and test results')
    p.add_argument('--seed',
                   type=int,
                   default=1,
                   help='random seed for pytorch')
    p.add_argument('--profiler',
                   type=str2bool,
                   default=False)

    # dist settings
    p.add_argument('--world-size',
                   type=int,
                   default=1,
                   help='total number of processes')
    p.add_argument('--ddp',
                   type=str2bool,
                   default=False,
                   help='whether to use DDP model')

    # feeder
    p.add_argument('--feeder',
                   default='feeder.feeder',
                   help='data loader will be used')
    p.add_argument('--num-worker',
                   type=int,
                   default=32,
                   help='the number of worker for data loader')
    p.add_argument('--train-feeder-args',
                   default=dict(),
                   help='the arguments of data loader for training')
    p.add_argument('--test-feeder-args',
                   default=dict(),
                   help='the arguments of data loader for test')
    p.add_argument('--train-dataloader-args',
                   default=dict(),
                   help='the arguments of data loader for training')
    p.add_argument('--test-dataloader-args',
                   default=dict(),
                   help='the arguments of data loader for test')
    p.add_argument('--use-sgn-dataloader',
                   type=str2bool,
                   default=False,
                   help='whether to use collate_fn from SGN')

    # model
    p.add_argument('--model',
                   default=None,
                   help='the model will be used')
    p.add_argument('--model-args',
                   type=dict,
                   default=dict(),
                   help='the arguments of model')
    p.add_argument('--weights',
                   default=None,
                   help='the weights for network initialization')
    p.add_argument('--ignore-weights',
                   type=str,
                   default=[],
                   nargs='+',
                   help='name of weights to be ignored in the initialization')
    p.add_argument('--label-smoothing', default=0.0)

    # optim
    p.add_argument('--start-epoch',
                   type=int,
                   default=0,
                   help='start training from which epoch')
    p.add_argument('--num-epoch',
                   type=int,
                   default=80,
                   help='stop training in which epoch')
    p.add_argument('--base-lr',
                   type=float,
                   default=0.01,
                   help='initial learning rate')
    p.add_argument('--step',
                   type=int,
                   default=[20, 40, 60],
                   nargs='+',
                   help='the epoch where optimizer reduce the learning rate')
    p.add_argument('--optimizer',
                   default='SGD',
                   help='type of optimizer')
    p.add_argument('--nesterov',
                   type=str2bool,
                   default=False,
                   help='use nesterov or not')
    p.add_argument('--weight-decay',
                   type=float,
                   default=0.0005,
                   help='weight decay for optimizer')
    p.add_argument('--llrd-factor',
                   type=float,
                   default=0.5,
                   help='factor for layerwise lr decay')
    p.add_argument('--eps',
                   type=float,
                   default=1e-8,
                   help='for adamw')
    p.add_argument('--only_train_part',
                   type=str2bool,
                   default=False,
                   help='if true skips the training of PA')
    p.add_argument('--only_train_epoch',
                   type=int,
                   default=0,
                   help='number of epochs for only_train_part')
    p.add_argument('--warm_up_epoch',
                   type=int,
                   default=0,
                   help='number of warmup epochs')

    # scheduler
    p.add_argument('--scheduler',
                   type=str,
                   default='',
                   help='pytorch lr scheduler')
    p.add_argument('--anneal-strategy',
                   type=str,
                   default='cos',
                   help='annealing strategy for pytorch lr scheduler')
    p.add_argument('--initial_lr',
                   type=float,
                   default=1e-5,
                   help='pytorch lr scheduler initial lr for cyclic scheduler')
    p.add_argument('--final_lr',
                   type=float,
                   default=1e-6,
                   help='pytorch lr scheduler final lr for cyclic scheduler')

    # processor
    p.add_argument('--batch-size',
                   type=int,
                   default=256,
                   help='training batch size')
    p.add_argument('--test-batch-size',
                   type=int,
                   default=256,
                   help='test batch size')
    p.add_argument('--device',
                   type=int,
                   default=0,
                   nargs='+',
                   help='the indexes of GPUs for training or testing')
    p.add_argument('--phase',
                   default='train',
                   help='must be train or test')
    p.add_argument('--save-score',
                   type=str2bool,
                   default=False,
                   help='if ture, the classification score will be stored')
    p.add_argument('--log-interval',
                   type=int,
                   default=100,
                   help='the interval for printing messages (#iteration)')
    p.add_argument('--save-interval',
                   type=int,
                   default=2,
                   help='the interval for storing models (#iteration)')
    p.add_argument('--eval-interval',
                   type=int,
                   default=5,
                   help='the interval for evaluating models (#iteration)')
    p.add_argument('--print-log',
                   type=str2bool,
                   default=True,
                   help='print logging or not')
    p.add_argument('--show-topk',
                   type=int,
                   default=[1, 5],
                   nargs='+',
                   help='which Top K accuracy will be shown')

    return p
