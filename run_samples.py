import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
import model
from model.tPatchGNN import *
import heapq

parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

parser.add_argument('--no_patch_ts', action='store_true', help='remove patch the time sequence.')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)


class Inspector:

    @staticmethod
    def load_ckpt(ckpt_path, device=None):
        if not os.path.exists(ckpt_path):
            raise Exception("Checkpoint " + ckpt_path + " does not exist.")
        # Load checkpoint.
        checkpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
        ckpt_args = checkpt['args']
        state_dict = checkpt['state_dicts']


        model = tPatchGNN(ckpt_args)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict) 
        # 3. load the new state dict
        model.load_state_dict(state_dict)

        if not device:
            device = ckpt_args.device 
        model.to(device)
        return model, ckpt_args
	
def infer(model, batch_dict):
	pred_y = model.forecasting(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			batch_dict["observed_mask"]) 
	return pred_y


#####################################################################################################
def evaluation_eval(model, dataloader, n_batches):
	
	# Retrieve the top 10 batches with the lowest MSE loss
	n_eval_samples = 0
	n_eval_samples_mape = 0
	total_results = {}
	total_results["loss"] = 0
	total_results["mse"] = 0
	total_results["mae"] = 0
	total_results["rmse"] = 0
	total_results["mape"] = 0

	for _ in range(n_batches):
		batch_dict = utils.get_next_batch(dataloader)
		if batch_dict is None:
			continue
		pred_y = model.forecasting(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			batch_dict["observed_mask"]) 
		
		# print('consistency test:', batch_dict["data_to_predict"][batch_dict["mask_predicted_data"].bool()].sum(), batch_dict["mask_predicted_data"].sum()) # consistency test
		
		# (n_dim, ) , (n_dim, ) 
		se_var_sum, mask_count = compute_error(batch_dict["data_to_predict"], pred_y, mask=batch_dict["mask_predicted_data"], func="MSE", reduce="sum") # a vector

		ae_var_sum, _ = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="sum") # a vector

		# norm_dict = {"data_max": batch_dict["data_max"], "data_min": batch_dict["data_min"]}
		ape_var_sum, mask_count_mape = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAPE", reduce="sum") # a vector


		# add a tensor (n_dim, )
		total_results["loss"] += se_var_sum
		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum
		total_results["mape"] += ape_var_sum
		n_eval_samples += mask_count
		n_eval_samples_mape += mask_count_mape


	n_avai_var = torch.count_nonzero(n_eval_samples)
	n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)
		
		### 1. Compute avg error of each variable first
		### 2. Compute avg error along the variables 
	total_results["loss"] = (total_results["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["rmse"] = torch.sqrt(total_results["mse"])
	total_results["mape"] = (total_results["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape



	for key, var in total_results.items(): 
		if isinstance(var, torch.Tensor):
			var = var.item()
		total_results[key] = var

	return total_results



if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	assert experimentID is not None, "Please specify the experiment ID to load the model."

	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	if not os.path.exists("experiments/"):
		utils.makedirs("experiments/")
	print("Experiment ID: {}".format(experimentID))

	model, args = Inspector.load_ckpt(ckpt_path, args.device)
	args.batch_size = 1

	data_obj = parse_datasets(args, patch_ts=True)
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	args.ndim = input_dim
	# model = getattr(getattr(model, args.model), args.model)(args).to(args.device)
	# model.load_state_dict(torch.load(ckpt_path))
	print(model)

	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	
	# model = tPatchGNN(args).to(args.device)

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log". \
			format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr)
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)


	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	best_val_mse = np.inf
	test_res = None
	st = time.time()
	model.eval()
	with torch.no_grad():
		val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
		test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

		print("val_res:", val_res)
		print("test_res:", test_res)
		
			

