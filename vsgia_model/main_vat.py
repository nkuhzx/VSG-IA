import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

import shutil
import sys
import random
import time
import numpy as np
from datetime import datetime

from vsgia_model.config import cfg
from vsgia_model.dataset.videotargetatt import VideoTargetAttLoader

from vsgia_model.utils.model_utils import init_model,setup_model,save_checkpoint,resume_checkpoint,init_checkpoint

from vsgia_model.tester_vat import Tester
# from tester_vat import Tester
# from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_engine(opt):

    # init gaze model
    gazemodel=init_model(opt)

    # set criterion and optimizer for gaze model
    criterion,optimizer=setup_model(gazemodel,opt)



    random.seed(opt.OTHER.seed)
    np.random.seed(opt.OTHER.seed)
    torch.manual_seed(opt.OTHER.seed)

    cudnn.deterministic=True

    # load the model weights
    if os.path.isfile(opt.TEST.model_para):

        gazemodel.load_state_dict(torch.load(opt.TEST.model_para))

    else:
        raise Exception("No such model file")


    dataloader = VideoTargetAttLoader(opt)
    val_loader=dataloader.val_loader

    # init trainer and validator for gazemodel
    tester=Tester(gazemodel,criterion,val_loader,opt,writer=None)

    # no require grad
    with torch.no_grad():
        eval_dist,eval_auc,eval_ap = tester.test(0,opt)


    print("Eval L2 dist.: {} | Eval AUC.:{} | Eval AP :{}".format(eval_dist,eval_auc,eval_ap))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch Attention Model"
    )

    parser.add_argument(
        "--cfg",
        default="config/videotargetattention.yaml",
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