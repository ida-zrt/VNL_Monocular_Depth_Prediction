from data.load_dataset import CustomerDataLoader
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_err_kitti
from lib.models.metric_depth_model import *
from lib.core.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_logging, SmoothedValue
import traceback
import math
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions
logger = setup_logging(__name__)

if __name__ == '__main__':

    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()
    train_opt.print_options(train_args)

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0
    val_opt.print_options(val_args)

    train_dataloader = CustomerDataLoader(train_args)

    for data in train_dataloader:
        print(type(data))
