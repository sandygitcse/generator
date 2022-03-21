from pdb import set_trace
import sys
import os
import argparse
import numpy as np
import torch
from models.base_models import EncoderRNN, DecoderRNN, Net_GRU, NetFullyConnected, get_base_model
from models.index_models import get_index_model
from loss.dilate_loss import dilate_loss
from train import train_model, get_optimizer
from eval import eval_base_model, eval_inf_model, eval_inf_index_model, eval_aggregates
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
import json
from torch.utils.tensorboard import SummaryWriter
import shutil
import properscoring as ps
import scipy.stats
import itertools

from functools import partial
torch.backends.cudnn.deterministic = True
from models import inf_models, inf_index_models
import utils

os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "1000000"

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, help='dataset_name')
#parser.add_argument('model_name', type=str, help='model_name')

parser.add_argument('--N_input', type=int, default=-1,
                    help='number of input steps')
parser.add_argument('--N_output', type=int, default=-1,
                    help='number of output steps')

parser.add_argument('--output_dir', type=str,
                    help='Path to store all raw outputs', default=None)
parser.add_argument('--saved_models_dir', type=str,
                    help='Path to store all saved models', default=None)

parser.add_argument('--ignore_ckpt', action='store_true', default=False,
                    help='Start the training without loading the checkpoint')

parser.add_argument('--normalize', type=str, default=None,
                    choices=['same', 'zscore_per_series', 'gaussian_copula', 'log'],
                    help='Normalization type (avg, avg_per_series, quantile90, std)')
parser.add_argument('--epochs', type=int, default=-1,
                    help='number of training epochs')

parser.add_argument('--print_every', type=int, default=50,
                    help='Print test output after every print_every epochs')

parser.add_argument('--learning_rate', type=float, default=-1.,# nargs='+',
                   help='Learning rate for the training algorithm')
parser.add_argument('--hidden_size', type=int, default=-1,# nargs='+',
                   help='Number of units in the encoder/decoder state of the model')
parser.add_argument('--num_grulstm_layers', type=int, default=-1,# nargs='+',
                   help='Number of layers of the model')

parser.add_argument('--fc_units', type=int, default=16, #nargs='+',
                   help='Number of fully connected units on top of the encoder/decoder state of the model')

parser.add_argument('--batch_size', type=int, default=-1,
                    help='Input batch size')

parser.add_argument('--gamma', type=float, default=0.01, nargs='+',
                   help='gamma parameter of DILATE loss')
parser.add_argument('--alpha', type=float, default=0.5,
                   help='alpha parameter of DILATE loss')
parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                   help='Probability of applying teacher forcing to a batch')
parser.add_argument('--deep_std', action='store_true', default=False,
                    help='Extra layers for prediction of standard deviation')
parser.add_argument('--train_twostage', action='store_true', default=False,
                    help='Train base model in two stages -- train only \
                          mean in first stage, train both in second stage')
parser.add_argument('--mse_loss_with_nll', action='store_true', default=False,
                    help='Add extra mse_loss when training with nll')
parser.add_argument('--second_moment', action='store_true', default=False,
                    help='compute std as std = second_moment - mean')
parser.add_argument('--variance_rnn', action='store_true', default=False,
                    help='Use second RNN to compute variance or variance related values')
parser.add_argument('--input_dropout', type=float, default=0.0,
                    help='Dropout on input layer')

parser.add_argument('--v_dim', type=int, default=-1,
                   help='Dimension of V vector in LowRankGaussian')
parser.add_argument('--b', type=int, default=-1,
                   help='Number of correlation terms to sample for loss computation during training')

#parser.add_argument('--use_feats', action='store_true', default=False,
#                    help='Use time features derived from calendar-date and other covariates')
parser.add_argument('--use_feats', type=int, default=-1,
                    help='Use time features derived from calendar-date and other covariates')

parser.add_argument('--t2v_type', type=str,
                    choices=['local', 'idx', 'mdh_lincomb', 'mdh_parti'],
                    help='time2vec type', default=None)

parser.add_argument('--use_coeffs', action='store_true', default=False,
                    help='Use coefficients obtained by decomposition, wavelet, etc..')


# Hierarchical model arguments
parser.add_argument('--L', type=int, default=2,
                    help='number of levels in the hierarchy, leaves inclusive')

parser.add_argument('--K_list', type=int, nargs='*', default=[],
                    help='List of bin sizes of each aggregation')

parser.add_argument('--wavelet_levels', type=int, default=2,
                    help='number of levels of wavelet coefficients')
parser.add_argument('--fully_connected_agg_model', action='store_true', default=False,
                    help='If True, aggregate model will be a feed-forward network')
parser.add_argument('--transformer_agg_model', action='store_true', default=False,
                    help='If True, aggregate model will be a Transformer')
parser.add_argument('--plot_anecdotes', action='store_true', default=False,
                    help='Plot the comparison of various methods')
parser.add_argument('--save_agg_preds', action='store_true', default=False,
                    help='Save inputs, targets, and predictions of aggregate base models')

parser.add_argument('--device', type=str,
                    help='Device to run on', default=None)

# parameters for ablation study
parser.add_argument('--leak_agg_targets', action='store_true', default=False,
                    help='If True, aggregate targets are leaked to inference models')
parser.add_argument('--patience', type=int, default=50,
                    help='Stop the training if no improvement shown for these many \
                          consecutive steps.')
#parser.add_argument('--seed', type=int,
#                    help='Seed for parameter initialization',
#                    default=42)

# Parameters for ARTransformerModel
parser.add_argument('--kernel_size', type=int, default=10,
                    help='Kernel Size of Conv (in ARTransformerModel)')
parser.add_argument('--nkernel', type=int, default=32,
                    help='Number of kernels of Conv (in ARTransformerModel)')
parser.add_argument('--dim_ff', type=int, default=512,
                    help='Dimension of Feedforward (in ARTransformerModel)')
parser.add_argument('--nhead', type=int, default=4,
                    help='Number of attention heads (in ARTransformerModel)')


# Cross-validation parameters
parser.add_argument('--cv_inf', type=int, default=-1,
                    help='Cross-validate the Inference models based on score on dev data')

# Learning rate for Inference Model
parser.add_argument('--lr_inf', type=float, default=-1.,
                    help='Learning rate for SGD-based inference model')

parser.add_argument('--initialization', type=float, default=-1.,
                    help='=1 for training median prediction model')



args = parser.parse_args()

#args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.base_model_names = [
#    'seq2seqdilate',
#    'seq2seqnll',
#    'seq2seqmse',
#    'convmse',
#    'convmsenonar',
#    'convnll',
#    'rnn-aggnll-nar',
#    'rnn-q-nar',
#    'rnn-q-ar',
#    'trans-mse-nar',
#    'trans-q-nar',
#    'nbeats-mse-nar',
#    'nbeatsd-mse-nar'
#    'rnn-mse-ar',
#    'rnn-nll-ar',
   'trans-mse-ar',
     'trans-huber-ar',
    'trans-nll-ar',
#    'trans-bvnll-ar',
#    'trans-nll-atr',
#    'trans-fnll-ar',
#    'rnn-mse-nar',
#    'rnn-nll-nar',
#    'rnn-fnll-nar',
#    'transm-nll-nar',
#    'transm-fnll-nar',
#    'transda-nll-nar',
#    'transda-fnll-nar',
#    'oracle',
#    'oracleforecast'
#    'transsig-nll-nar',
]
args.aggregate_methods = ['sum']
args.K_list = [1]

args.inference_model_names = []
if 'seq2seqdilate' in args.base_model_names:
    args.inference_model_names.append('DILATE')
if 'seq2seqmse' in args.base_model_names:
    args.inference_model_names.append('MSE')
if 'seq2seqnll' in args.base_model_names:
    args.inference_model_names.append('NLL')
if 'convmse' in args.base_model_names:
    args.inference_model_names.append('CNNRNN-MSE')
if 'convnll' in args.base_model_names:
    args.inference_model_names.append('CNNRNN-NLL')
if 'convmsenonar' in args.base_model_names:
    args.inference_model_names.append('CNNRNN-NONAR-MSE')
if 'rnn-aggnll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-AGGNLL-NAR')
if 'rnn-q-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-Q-NAR')
if 'rnn-mse-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-MSE-AR')
if 'rnn-q-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-Q-AR')
if 'trans-mse-nar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-NAR')
if 'trans-q-nar' in args.base_model_names:
    args.inference_model_names.append('TRANS-Q-NAR')
if 'nbeats-mse-nar' in args.base_model_names:
    args.inference_model_names.append('NBEATS-MSE-NAR')
if 'nbeatsd-mse-nar' in args.base_model_names:
    args.inference_model_names.append('NBEATSD-MSE-NAR')
if 'rnn-mse-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-MSE-NAR')
if 'rnn-nll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-NLL-NAR')
if 'rnn-nll-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-NLL-AR')
if 'trans-mse-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-AR')
if 'trans-huber-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-HUBER-AR')
if 'trans-nll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-AR')
if 'trans-bvnll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-BVNLL-AR')
if 'trans-nll-atr' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-ATR')
if 'trans-fnll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-FNLL-AR')
if 'rnn-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-FNLL-NAR')
if 'transm-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSM-NLL-NAR')
if 'transm-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSM-FNLL-NAR')
if 'transda-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSDA-NLL-NAR')
if 'transda-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSDA-FNLL-NAR')
if 'oracle' in args.base_model_names:
    args.inference_model_names.append('oracle')
if 'oracleforecast' in args.base_model_names:
    args.inference_model_names.append('SimRetrieval')
if 'transsig-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSSIG-NLL-NAR')




if args.dataset_name in ['Traffic']:
    args.alpha = 0.8

if args.dataset_name in ['ECG5000']:
    args.teacher_forcing_ratio = 0.0

if args.dataset_name in ['Solar']:
    opt_normspace = False
else:
    opt_normspace = True

#import ipdb ; ipdb.set_trace()
if args.dataset_name == 'ett':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 192
    if args.N_output == -1: args.N_output = 192
    if args.K_list == []: args.K_list = []
    #args.K_list = [6]
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_ett_d192_b24_e192_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2v_usefeats_t2vglobal_idx_val20'
    if args.output_dir is None:
        args.output_dir = 'Outputs_ett_d192_klnorm_b24_e192_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2v_usefeats_t2vglobal_idx_val20'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.00001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size  == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.t2v_type = 'idx'
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    #python main.py ett --epochs 20 --N_input 192 --N_output 192 --K_list 6 --saved_models_dir saved_models_ett_d192 --output_dir Outputs_ett_d192_klnorm --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --device cuda:0

elif args.dataset_name == 'taxi30min':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_taxi30min_d168_b48_pefix_e336_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2vglobal_mdh_parti'
    if args.output_dir is None:
        args.output_dir = 'Outputs_taxi30min_d168_klnorm_b48_pefix_e336_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2vglobal_mdh_parti'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 128
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    #args.t2v_type = 'mdh_parti'
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01

elif args.dataset_name == 'etthourly':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 168
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_etthourly_noextrafeats_d168_b24_pefix_e168_val20_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.output_dir is None:
        args.output_dir = 'Outputs_etthourly_noextrafeats_d168_klnorm_b24_pefix_e168_val20_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.00001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.print_every = 5 # TODO: Only for aggregate models
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01

elif args.dataset_name == 'azure':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 720
    if args.N_output == -1: args.N_output = 360
    #args.K_list = [60]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_azure_d360_e720_usefeats_bs128_normsame'
    if args.output_dir is None:   
        args.output_dir = 'Outputs_azure_d360_e720_usefeats_bs128_normsame'
    #args.normalize = 'zscore_per_series'
    if args.normalize is None: args.normalize = 'same'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 128
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 10
    if args.use_feats == -1: args.use_feats = 1
    #args.t2v_type = None
    if args.device is None: args.device = 'cuda:0'
    if args.cv_inf == -1: args.cv_inf = 1

elif args.dataset_name == 'Solar':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_Solar_d168_b4_e336_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.output_dir is None:
        args.output_dir = 'Outputs_Solar_d168_normzscore_klnorm_b4_e336_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.001

elif args.dataset_name == 'electricity':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_electricity'
    if args.output_dir is None:
        args.output_dir = 'Outputs_electricity'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 128
    if args.hidden_size == -1: args.hidden_size = 256
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01

elif args.dataset_name == 'aggtest':
    if args.epochs == -1: args.epochs = 1
    if args.N_input == -1: args.N_input = 20
    if args.N_output == -1: args.N_output = 10
    #args.K_list = [12]
    if args.K_list == []: args.K_list = [1, 5]
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_aggtest_test'
    if args.output_dir is None:
        args.output_dir = 'Outputs_aggtest_test'
    if args.normalize is None: args.normalize = 'same'
    if args.learning_rate == -1.: args.learning_rate = 0.001
    if args.batch_size == -1: args.batch_size = 100
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = args.N_output
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1


elif args.dataset_name == 'Traffic911':
    args.epochs = 20
    args.N_input = 336
    args.N_output = 168
    args.K_list = [6]
    args.saved_models_dir = 'saved_models_Traffic911_d168'
    args.output_dir = 'Outputs_Traffic911_d168'
    args.normalize = 'zscore_per_series'
    args.learning_rate = 0.0001
    args.batch_size = 128
    args.hidden_size = 128
    args.num_grulstm_layers = 1
    args.v_dim = 1
    args.print_every = 5 # TODO: Only for aggregate models
    args.device = 'cuda:0'

elif args.dataset_name == 'foodinflation':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 90
    if args.N_output == -1: args.N_output = 30
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_foodinflation'
    if args.output_dir is None:
        args.output_dir = 'Outputs_foodinflation'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01


elif args.dataset_name == 'outlier':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 100
    if args.N_output == -1: args.N_output = 50
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models/saved_models_outlier'
    if args.output_dir is None:
        args.output_dir = 'Outputs/Outputs_foodinflation'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01

elif args.dataset_name == 'telemetry':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 90
    if args.N_output == -1: args.N_output = 30
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_telemetry'
    if args.output_dir is None:
        args.output_dir = 'Outputs_telemetry'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.initialization == -1: args.initialization = 1.

print('Command Line Arguments:')
print(args)

#import ipdb ; ipdb.set_trace()

base_models = {}
base_models_preds = {}
for name in args.base_model_names:
    base_models[name] = {}
    base_models_preds[name] = {}
inference_models = {}
for name in args.inference_model_names:
    inference_models[name] = {}


# DUMP_PATH = '/mnt/infonas/data/pratham/Forecasting/DILATE'
DUMP_PATH = '.'
args.output_dir = os.path.join(DUMP_PATH, args.output_dir)
args.saved_models_dir = os.path.join(DUMP_PATH, args.saved_models_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.saved_models_dir, exist_ok=True)



#dataset = utils.get_processed_data(args)
data_processor = utils.DataProcessor(args)
#level2data = dataset['level2data']

# ----- Start: Load all datasets ----- #

dataset = {}
agg_method = args.aggregate_methods[0]
dataset[agg_method] = {}
level = args.K_list[0]
dataset[agg_method][level] = data_processor.get_processed_data(args, agg_method, level)

# ----- End : Load all datasets ----- #

# ----- Start: base models training ----- #
# set_trace()
for base_model_name in args.base_model_names:
    base_models[base_model_name] = {}
    base_models_preds[base_model_name] = {}

    level = args.K_list[0]
    agg_method = args.aggregate_methods[0]

    base_models[base_model_name][agg_method] = {}
    base_models_preds[base_model_name][agg_method] = {}

    level2data = dataset[agg_method][level]
    trainloader = level2data['trainloader']
    devloader = level2data['devloader']
    testloader = level2data['testloader']
    feats_info = level2data['feats_info']
    N_input = level2data['N_input']
    N_output = level2data['N_output']
    input_size = level2data['input_size']
    output_size = level2data['output_size']
    dev_norm = level2data['dev_norm']
    test_norm = level2data['test_norm']
    
    if base_model_name in [
        'seq2seqmse', 'seq2seqdilate', 'convmse', 'convmsenonar',
        'rnn-mse-nar', 'rnn-mse-ar', 'trans-mse-nar', 'nbeats-mse-nar',
        'nbeatsd-mse-nar', 'trans-mse-ar', 'oracle', 'oracleforecast','trans-huber-ar'
    ]:
        estimate_type = 'point'
    elif base_model_name in [
        'seq2seqnll', 'convnll', 'trans-q-nar', 'rnn-q-nar', 'rnn-q-ar',
        'rnn-nll-nar', 'rnn-nll-ar', 'rnn-aggnll-nar', 'trans-nll-ar',
        'transm-nll-nar', 'transda-nll-nar', 'transsig-nll-nar', 'trans-nll-atr'
    ]:
        estimate_type = 'variance'
    elif base_model_name in [
        'rnn-fnll-nar', 'trans-fnll-ar', 'transm-nll-nar', 'transda-fnll-nar'
    ]:
        estimate_type = 'covariance'
    elif base_model_name in ['trans-bvnll-ar']:
        estimate_type = 'bivariate'

    saved_models_dir = os.path.join(
        args.saved_models_dir,
        args.dataset_name+'_'+base_model_name+'_'+agg_method+'_'+str(level)
    )
    os.makedirs(saved_models_dir, exist_ok=True)
    writer = SummaryWriter(saved_models_dir)
    saved_models_path = os.path.join(saved_models_dir, 'state_dict_model.pt')
    print('\n {} {} {}'.format(base_model_name, agg_method, str(level)))


    # Create the network
    # import pdb;pdb.set_trace()
    net_gru = get_base_model(
        args, base_model_name, level,
        N_input, N_output, input_size, output_size,
        estimate_type, feats_info
    )

    # train the network
    if agg_method in ['sumwithtrend', 'slope', 'wavelet', 'haar'] and level == 1:
        base_models[base_model_name][agg_method][level] = base_models[base_model_name]['sum'][1]
    else:
        # import pdb;pdb.set_trace()
        if base_model_name not in ['oracle', 'oracleforecast']:
            train_model(
                args, base_model_name, net_gru,
                level2data, saved_models_path, writer, verbose=1
            )

        base_models[base_model_name][agg_method][level] = net_gru

    writer.flush()


writer.close()
            #import ipdb
            #ipdb.set_trace()
# ----- End: base models training ----- #

# ----- Start: Inference models for bottom level----- #
print('\n Starting Inference Models')

#import ipdb
#ipdb.set_trace()


def run_inference_model(
    args, inf_model_name, base_models, which_split, opt_normspace, agg_method=None, K=None
):

    metric2val = dict()
    infmodel2preds = dict()

    if inf_model_name in ['DILATE']:
        base_models_dict = base_models['seq2seqdilate']['sum']
        inf_net = inf_models.DILATE(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-MSE-NAR']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-NLL-NAR']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-NLL-AR']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-MSE-AR']:
        base_models_dict = base_models['trans-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-HUBER-AR']:
        base_models_dict = base_models['trans-huber-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-NLL-AR']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-BVNLL-AR']:
        base_models_dict = base_models['trans-bvnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-NLL-ATR']:
        base_models_dict = base_models['trans-nll-atr']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-FNLL-AR']:
        base_models_dict = base_models['trans-fnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-FNLL-NAR']:
        base_models_dict = base_models['rnn-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['oracle']:
        base_models_dict = base_models['oracle']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device, is_oracle=True)

    elif inf_model_name in ['SimRetrieval']:
        base_models_dict = base_models['oracleforecast']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device, is_oracle=True)

    elif inf_model_name in ['TRANSSIG-NLL-NAR']:
        base_models_dict = base_models['transsig-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSM-NLL-NAR']:
        base_models_dict = base_models['transm-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSM-FNLL-NAR']:
        base_models_dict = base_models['transm-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSDA-NLL-NAR']:
        base_models_dict = base_models['transda-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSDA-FNLL-NAR']:
        base_models_dict = base_models['transda-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-AGGNLL-NAR']:
        base_models_dict = base_models['rnn-aggnll-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-Q-NAR']:
        base_models_dict = base_models['rnn-q-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['RNN-MSE-AR']:
        base_models_dict = base_models['rnn-mse-ar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['RNN-Q-AR']:
        base_models_dict = base_models['rnn-q-ar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['TRANS-MSE-NAR']:
        base_models_dict = base_models['trans-mse-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['TRANS-Q-NAR']:
        base_models_dict = base_models['trans-q-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['NBEATS-MSE-NAR']:
        base_models_dict = base_models['nbeats-mse-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['NBEATSD-MSE-NAR']:
        base_models_dict = base_models['nbeatsd-mse-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    if not args.leak_agg_targets:
        inf_test_targets_dict = None

    inf_net.eval()
    (
        inputs, target, pred_mu, pred_std, pred_d, pred_v,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape,
        total_time
    ) = eval_inf_model(args, inf_net, dataset, which_split, args.gamma, verbose=1)

    if inf_net.covariance == False:
        pred_v_foragg = None
    else:
        pred_v_foragg = pred_v
    #import ipdb ; ipdb.set_trace()
    agg2metrics = eval_aggregates(
        inputs, target, pred_mu, pred_std, pred_d, pred_v_foragg
    )

    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('Metrics for Inference model {}: MAE:{:f}, CRPS:{:f}, MSE:{:f}, SMAPE:{:f}, Time:{:f}'.format(
        inf_model_name, metric_mae, metric_crps, metric_mse, metric_smape, total_time)
    )

    metric2val = utils.add_metrics_to_dict(
        metric2val, inf_model_name,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape
    )
    infmodel2preds[inf_model_name] = pred_mu
    if which_split in ['test']:
        output_dir = os.path.join(args.output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        utils.write_arr_to_file(
            output_dir, inf_model_name,
            inputs.detach().numpy(),
            target.detach().numpy(),
            pred_mu.detach().numpy(),
            pred_std.detach().numpy(),
            pred_d.detach().numpy(),
            pred_v.detach().numpy()
        )

    return metric2val, agg2metrics

model2metrics = dict()
model2aggmetrics = dict()
for inf_model_name in args.inference_model_names:

    if args.cv_inf:
        # Consider all possible combinations of aggregate methods
        aggregate_methods = []
        for l in range(1, len(args.aggregate_methods)+1):
            aggregate_methods += list(itertools.combinations(args.aggregate_methods, l))
        # Single value of K is used in a hyper-parameter config for inference model
        K_list = []
        if len(args.K_list) == 1:
            K_list = [args.K_list]
        else:
            for K in args.K_list:
                if K != 1:
                    K_list.append([1, K])

        hparam_configs = list(itertools.product(aggregate_methods, K_list))

        hparams2metrics = []
        for agg_method, K in hparam_configs:
            print('cv with agg_method, K:', agg_method, K)
            metric2val, agg2metrics = run_inference_model(
                args, inf_model_name, base_models, 'dev', opt_normspace,
                agg_method, K
            )
            hparams2metrics.append(metric2val)
        cv_metric =  'crps'
        best_cfg_idx, _ = min(enumerate(hparams2metrics), key=lambda x: x[1]['crps'])
        print(
            'best_cfg_idx:', best_cfg_idx,
            'best_agg_method and best K:', hparam_configs[best_cfg_idx],
        )
        metric2val, agg2metrics = run_inference_model(
            args, inf_model_name, base_models, 'test', opt_normspace,
            hparam_configs[best_cfg_idx][0],
            hparam_configs[best_cfg_idx][1]
        )
        model2metrics[inf_model_name] = metric2val
        model2aggmetrics[inf_model_name] = agg2metrics
    else:
        #raise NotImplementedError
        metric2val, agg2metrics = run_inference_model(
            args, inf_model_name, base_models, 'test', opt_normspace
        )
        model2metrics[inf_model_name] = metric2val
        model2aggmetrics[inf_model_name] = agg2metrics


with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.txt'), 'w') as fp:

    fp.write('\nModel Name, MAE, DTW, TDI')
    for model_name, metrics_dict in model2metrics.items():
        fp.write(
            '\n{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                model_name,
                metrics_dict['mae'],
                metrics_dict['crps'],
                metrics_dict['mse'],
                metrics_dict['dtw'],
                metrics_dict['tdi'],
            )
        )

for model_name, metrics_dict in model2metrics.items():
    for metric, metric_val in metrics_dict.items():
        model2metrics[model_name][metric] = str(metric_val)
with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.json'), 'w') as fp:
    json.dump(model2metrics, fp)

# ----- End: Inference models for bottom level----- #