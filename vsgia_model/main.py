import sys
import os

# print(os.path.abspath(os.path.join(os.getcwd(),".")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

import torch
import torch.backends.cudnn as cudnn
import argparse

import shutil
import sys
import random
import time
import numpy as np
from datetime import datetime

from vsgia_model.config import cfg
from vsgia_model.dataset.gazefollow import GazeFollowLoader

from vsgia_model.utils.model_utils import init_model,setup_model,save_checkpoint,resume_checkpoint,init_checkpoint

from vsgia_model.tester import Tester






def test_engine(opt):

    # init model
    model=init_model(opt)

    # set criterion and optimizer for gaze model
    criterion,optimizer=setup_model(model,opt)

    random.seed(opt.OTHER.seed)
    np.random.seed(opt.OTHER.seed)
    torch.manual_seed(opt.OTHER.seed)

    cudnn.deterministic=True

    # load the model weights
    if os.path.isfile(opt.TEST.model_para):
        model.load_state_dict(torch.load(opt.TEST.model_para))

    else:
        raise Exception("No such model file")


    dataloader = GazeFollowLoader(opt)
    val_loader=dataloader.test_loader

    # init trainer and validator for gazemodel
    tester=Tester(model,criterion,val_loader,opt,writer=None)

    eval_dist,eval_mindist,eval_auc = tester.test(opt)

    print("Eval Avg dist.: {} | Eval Min dist.:{} | Eval AUC :{}".format(eval_dist,eval_mindist,eval_auc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch VSG-IA Model"
    )

    parser.add_argument(
        "--cfg",
        default="config/gazefollow_cfg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="choose if use gpus"
    )
    parser.add_argument(
        "--is_train",
        action="store_true",
        default=False,
        help="choose if train"
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        default=True,
        help="choose if test"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.OTHER.device='cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu'
    print("The model running on {}".format(cfg.OTHER.device))

    test_engine(cfg)




