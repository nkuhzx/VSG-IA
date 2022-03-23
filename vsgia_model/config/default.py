from yacs.config import CfgNode as CN
import os

def getRootPath():

    rootPath=os.path.dirname(os.path.abspath(__file__))


    rootPath=rootPath.split("/vsgia_model/config")[0]

    return rootPath
# -----------------------------------------------------------------------------
# Default Config definition
# -----------------------------------------------------------------------------

_C=CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASET=CN()

_C.DATASET.root_dir = os.path.join(getRootPath(),"datasets/gazefollow")
_C.DATASET.train_anno = os.path.join(getRootPath(),"datasets/gazefollow/train_annotation.txt")
_C.DATASET.test_anno = os.path.join(getRootPath(),"datasets/gazefollow/test_annotation.txt")

_C.DATASET.mask_dir=os.path.join(getRootPath(),"datasets/gazefollow_masks")

_C.DATASET.train_graph= os.path.join(getRootPath(),"datasets/gazefollow_graphinfo/train_graph_data.hdf5")

_C.DATASET.test_graph= os.path.join(getRootPath(),"datasets/gazefollow_graphinfo/test_graph_data.hdf5")

# dataset loader
_C.DATASET.load_workers=0
_C.DATASET.train_batch_size=32
_C.DATASET.test_batch_size=64

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL=CN()

_C.MODEL.inout_branch=False

# -----------------------------------------------------------------------------
# Transformer config
# -----------------------------------------------------------------------------
# Transformer parameters
_C.TRFM=CN()

_C.TRFM.hidden_dim=512
_C.TRFM.dropout=0.1
_C.TRFM.nheads=8
_C.TRFM.dim_feedforward=2048
_C.TRFM.enc_layers=6
_C.TRFM.dec_layers=6
_C.TRFM.pre_norm='None'
_C.TRFM.num_queries=16

# -----------------------------------------------------------------------------
# VSGAT config
# -----------------------------------------------------------------------------
# VSGAT parameters

_C.VSGAT=CN()

bias=True
dropout=0.2

_C.VSGAT.multi_attn=False

_C.VSGAT.diff_edge=True

_C.VSGAT.edge_apply_layers=[1024*2+18,1024]
_C.VSGAT.edge_apply_activate=['ReLU']
_C.VSGAT.edge_apply_bias=bias
_C.VSGAT.edge_apply_dropout=dropout

_C.VSGAT.edge_attn_layers=[1024,1]
_C.VSGAT.edge_attn_activate=['LeakyReLU']
_C.VSGAT.edge_attn_bias=bias
_C.VSGAT.edge_attn_dropout=dropout

_C.VSGAT.node_apply_layers=[2048,1024]
_C.VSGAT.node_apply_activate=['ReLU']
_C.VSGAT.node_apply_bias=bias
_C.VSGAT.node_apply_dropout=dropout


_C.VSGAT.pred_layers=[1024*2+18,1024,1]
_C.VSGAT.pred_activate=['ReLU','Sigmoid']
_C.VSGAT.pred_bias=bias
_C.VSGAT.pred_dropout=dropout


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

_C.TRAIN=CN()

# pre-trained parameters
_C.TRAIN.headpretrain=os.path.join(getRootPath(),"modelparas/gaze360pretrain.pt")
#_C.TRAIN.scenepretrain=""

_C.TRAIN.criterion="mixed"
_C.TRAIN.optimizer="adam"

_C.TRAIN.maxlr=3.5e-4
_C.TRAIN.minlr=0.000001
_C.TRAIN.weightDecay=1e-4
_C.TRAIN.epsilion=1e-8

_C.TRAIN.start_epoch=0
_C.TRAIN.end_epoch=40

# input and output resolution
_C.TRAIN.input_size=224
_C.TRAIN.output_size=64

# model parameters save interval and address
_C.TRAIN.store=os.path.join(getRootPath(),"modelparas/savemodel")
_C.TRAIN.save_intervel=1

# model parameters resume and initmodel
_C.TRAIN.resume=False
_C.TRAIN.resume_add=''

_C.TRAIN.initmodel=False
_C.TRAIN.initmodel_add=''

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

_C.TEST=CN()

_C.TEST.model_para=os.path.join(getRootPath(),"modelparas/model_gazefollow.pt")


# -----------------------------------------------------------------------------
# Other Default
# -----------------------------------------------------------------------------
_C.OTHER=CN()


_C.OTHER.seed=235
# if gpu is used
_C.OTHER.device='cpu'

# log for tensorboardx
_C.OTHER.logdir='../logs'

_C.OTHER.global_step=0

_C.OTHER.lossrec_every=10

_C.OTHER.evalrec_every=600




