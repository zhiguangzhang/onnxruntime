# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import math
from apex import amp
import multiprocessing

from tokenization import BertTokenizer
from modeling import BertForPreTraining, BertConfig
from optimization import BertLAMB
from utils import is_main_process
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from apex.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler
from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C
from apex.amp import _amp_state

from concurrent.futures import ProcessPoolExecutor

#############################################################
# BEGIN: ORT backend specific code block
#############################################################
# import onnx
import sys

import onnxruntime as ort

from onnxruntime.capi.ort_trainer import FuseSofmaxNLLToSoftmaxCE, create_ort_training_session_bind_parameters
from onnxruntime.capi.ort_trainer import ORTTrainer, ORTModel, IODescription, ModelDescription, create_ort_training_session_bind_parameters
from onnxruntime.capi.ort_trainer import ort_training_session_run_helper

from onnxruntime.capi.ort_trainer import LossScaler
from azureml_adapter import set_environment_variables_for_nccl_backend, get_local_rank, get_local_size, get_global_size, get_world_size, get_world_rank 

def ort_trainer_learning_rate_description():
    return IODescription('Learning_Rate', [1,], torch.float32)

def bert_model_description():
    vocab_size = 30528
    input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch',], torch.int64, num_classes = 2)
    loss_desc = IODescription('loss', [], torch.float32)
    # probability_desc = IODescription('probability', ['batch', 10], torch.float32)

    return ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc, next_sentence_labels_desc], [loss_desc])

def postprocess_model(model):
    # TODO: postprocess transforms are to be moved to either pytorch exporter or onnxruntime front end (ort_trainer.py)
    from ort_training_temp.model_transform import add_name, process_concat, handle_expand_input_is_not_constant_case, fix_expand, fix_dim, fix_transpose, process_dropout, add_expand_shape

    add_name(model)
    #replace garther&concat to reshape
    # import onnx
    # onnx.save(model, "/bert_ort/liqun/test_out/bert_tiny_before_process_concat.onnx")
    process_concat(model)

    # will be longer needed after Range is supported in ORT.
    handle_expand_input_is_not_constant_case(model)
    
    # fix the expand with dynamic shape
    # will be longer needed after Range is supported in ORT.
    fix_expand(model)

    #use dynamic batch/sequence
    fix_dim(model)

    #constant fold transpose
    #fix_transpose(model)

    #replace dropout with trainable dropout
    process_dropout(model)
    
    #add output shape of expand
    # will be longer needed after Range is supported in ORT.
    add_expand_shape(model)
    #set opset version to 10
    #model.opset_import[0].version = 10


    ################
    # layer_norm
    ################
    from ort_training_temp.layer_norm_transform import layer_norm_transform
    layer_norm_transform(model)

#############################################################
# END: ORT backend specific code block
#############################################################

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args):

    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, num_workers=0,
                                  pin_memory=True)
    return train_dataloader, input_file

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--accumulate_into_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use fp16 gradient accumulators.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--tensorboard_dir',
                        default=None,
                        type=str)
    parser.add_argument('--use_ort_trainer',
                        default=False,
                        action='store_true',
                        help="Whether to run with ort in fully optimized mode (run optimization in ort as opposed in pytorch).")
    parser.add_argument('--schedule',
                        default='warmup_poly',
                        type=str)
    parser.add_argument('--use_ort_trainer_nccl',
                        default=False,
                        action='store_true',
                        help="Whether to run with ort trainer with NCCL instead of Horovod.")
    parser.add_argument('--use_ib',
                        default=False,
                        help='Whether to enable IB or not')                    
    args = parser.parse_args()
    return args

def setup_training(args):

    assert (torch.cuda.is_available())
    has_aml = 'AZ_BATCH_MASTER_NODE' in os.environ.keys() or 'AZ_BATCHAPI_MPI_MASTER_NODE' in os.environ.keys()
    try:
        import mpi4py
        has_mpi4py = True
    except ImportError:
        has_mpi4py = False
        pass
    print ("has aml flag: " + str(has_aml))
    if has_aml:
        
        set_environment_variables_for_nccl_backend(get_local_size() == get_global_size(), IB = args.use_ib)

        args.local_rank = get_local_rank()
        args.world_rank = get_world_rank()
        args.world_size = get_global_size()
        if args.use_ort_trainer:
          torch.cuda.set_device(args.local_rank)
          device = torch.device("cuda", args.local_rank)
          args.n_gpu = 1
          from onnxruntime.capi._pybind_state import set_cuda_device_id 
          set_cuda_device_id(args.local_rank) 
        else:
          torch.cuda.set_device(args.local_rank) 
          device = torch.device("cuda", args.local_rank)
          args.n_gpu = 1
          torch.distributed.init_process_group(backend='nccl', init_method='env://')

    elif has_mpi4py and args.use_ort_trainer:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        # gpu_offset = 1
        # print("Warning: Using GPU offset of {}".format(gpu_offset))
        args.local_rank = comm.Get_rank() # + gpu_offset
        args.world_rank = comm.Get_rank()
        args.world_size=comm.Get_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

        from onnxruntime.capi._pybind_state import set_cuda_device_id 
        set_cuda_device_id(args.local_rank) 
    elif args.local_rank == -1:
        device = torch.device("cuda", 1)
        #we want to run single gpu
        args.n_gpu = 1
        args.world_size = 1
        # ort backend requires world_rank >= 0 && world_size > 0
        args.local_rank = 0 
        #args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("device %s n_gpu %d distributed training %r", device, args.n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not args.resume_from_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device):
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = BertForPreTraining(config)


    if args.use_ort_trainer:
        ##################################################
        # TODO: To condition on a command line argument as for
        # whether to run with ORT or PyTorch backend.

        # set GPU memory limitation
        from onnxruntime.capi._pybind_state import set_cuda_mem_limit
        ort_cuda_mem_limit_in_gbs = 15
        set_cuda_mem_limit(int(ort_cuda_mem_limit_in_gbs * 1024 * 1024 *1024))

        model_desc = bert_model_description()
        # we request ORTTrainer to create a LambOptimizer with given optimizer_attributes. 
        # train_step does forward, backward, and optimize step.
        learning_rate_description = ort_trainer_learning_rate_description()
            
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # args.local_rank = comm.Get_rank()

        # print("args.local_rank:", args.local_rank)
        # print("torch.cuda.device_count():", torch.cuda.device_count())

        def map_optimizer_attributes(name):
            no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
            no_decay = False
            for no_decay_key in no_decay_keys:
                if no_decay_key in name:
                    no_decay = True
                    break
            if no_decay:
                return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
            else:
                return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

        model = ORTTrainer(model, None, model_desc, "LambOptimizer", 
            map_optimizer_attributes,
            learning_rate_description,
            device, postprocess_model=postprocess_model, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,                
            # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
            world_rank=args.local_rank, world_size=args.world_size,
            use_mixed_precision = True if args.fp16 else False,
            allreduce_post_accumulation = True if args.allreduce_post_accumulation else False)
        ##################################################

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
        global_step = args.resume_step

        checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")

        model.load_state_dict(checkpoint['model'], strict=False)
        if args.phase2:
            global_step -= args.phase1_end_step
        
        if is_main_process(args):
            print("resume step from ", args.resume_step)

    if args.use_ort_trainer:
      return model, None, checkpoint, global_step
    
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    
    optimizer_grouped_parameters = []
    names = []

    count = 1
    for n, p in param_optimizer:
        count += 1
        if not any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({'params': [p], 'weight_decay': 0.01, 'name': n})
            names.append({'params': [n], 'weight_decay': 0.01})
        if any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({'params': [p], 'weight_decay': 0.00, 'name': n})
            names.append({'params': [n], 'weight_decay': 0.00})

    optimizer = BertLAMB(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=args.max_steps)

    if args.fp16:

        if args.loss_scale == 0:
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", 
                    master_weights=False if args.accumulate_into_fp16 else True)
        else:
            # optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale,
                    master_weights=False if args.accumulate_into_fp16 else True)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.resume_from_checkpoint:
        if args.phase2:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from Phase 1
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

        # Restore AMP master parameters          
        if args.fp16 and optimizer:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer, checkpoint, global_step

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0] if args.fp16 else None
        if args.fp16:
            master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        else:
            master_grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        loss_scale = 1.0 if scaler is None else scaler.loss_scale()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            loss_scale / (torch.distributed.get_world_size() * args.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        loss_unscale = 1.0 if scaler is None else 1./scaler.loss_scale()
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            loss_unscale)
        # 5. update loss scale
        if args.fp16:
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overfloat_buf = old_overflow_buf
        else:
            had_overflow = 0
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            if is_main_process(args):
                print(("Rank {} :: Gradient overflow.  Skipping step, "  +
                        "reducing loss scale to {}").format(
                        torch.distributed.get_rank(),
                        scaler.loss_scale()))
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.step()
        #optimizer.zero_grad()
        # TODO: We need to re-create grad and rebound parameters for this following code to work with ORT backend.
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step

def main():

    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, args = setup_training(args)

    # if args.use_ort_trainer:
    #     from train_with_ort_trainer import train_with_ort_trainer_hvd
    #     train_with_ort_trainer_hvd(args, logger)

    # Prepare optimizer
    model, optimizer, checkpoint, global_step = prepare_model_and_optimizer(args, device)

    if is_main_process(args):
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        print("SEED {}".format(args.seed))

    if args.do_train:
        if is_main_process(args):
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", args.train_batch_size)
            print("  LR = ", args.learning_rate)
            print("Training. . .")

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0

        pool = ProcessPoolExecutor(1)
        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            if not args.resume_from_checkpoint or epoch > 0 or args.phase2:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f))
                         and 'training' in f
                        ]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                f_names = [os.path.basename(_) for _ in files]
                files = [os.path.join(args.input_dir, f) for f in f_names]
                args.resume_from_checkpoint = False
                num_files = len(files)


            shared_file_list = {}

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            elif hasattr(args, 'world_size'):
                world_size = args.world_size
            else:
                world_size = 1

            if world_size > num_files:
                remainder = world_size % num_files
                data_file = files[(f_start_id*world_size + args.world_rank + remainder*f_start_id)%num_files]
            elif world_size > 1:
                data_file = files[(f_start_id*world_size + args.world_rank)%num_files]
            else:
                data_file = files[f_start_id % num_files]

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size * args.n_gpu, num_workers=0,
                                          pin_memory=True)
            # shared_file_list["0"] = (train_dataloader, data_file)

            overflow_buf = None
            if args.allreduce_post_accumulation and not args.use_ort_trainer:
                overflow_buf = torch.cuda.IntTensor([0])
            if args.fp16 and args.use_ort_trainer:
                loss_scaler = LossScaler(model.loss_scale_input_name, True, up_scale_window=2000)
            for f_id in range(f_start_id + 1 , len(files)):
                if world_size > num_files:
                    data_file = files[(f_id*world_size+args.world_rank + remainder*f_id)%num_files]
                elif world_size > 1:
                    data_file = files[(f_id*world_size + args.world_rank)%num_files]
                else:
                    data_file = files[f_id%num_files]

                logger.info("file no %s file %s" % (f_id, previous_file))

                previous_file = data_file

                dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args)

                train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process(args) else train_dataloader
                for step, batch in enumerate(train_iter):
                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    divisor = args.gradient_accumulation_steps
                    if args.use_ort_trainer:
                        from train_with_ort_trainer import get_lr
                        lr = get_lr(args, global_step, args.schedule)
                        learning_rate = torch.tensor([lr])
                        if args.fp16:
                            loss_scale = torch.tensor([loss_scaler.loss_scale_])
                            loss = model.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate, loss_scale)
                            all_finite = 1
                            if isinstance(loss, (list, tuple)):
                                assert len(loss) == 2
                                loss, all_finite = loss
                        else:
                            loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate)
                        if training_steps % args.gradient_accumulation_steps == 0:
                            if args.fp16:
                                loss_scaler.update_loss_scale(all_finite.item())
                            if is_main_process(args):
                                writer.add_scalar('train/summary/scalar/Learning_Rate', lr, 
                                    global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                                if args.fp16:
                                    writer.add_scalar('train/summary/scalar/loss_scale_25', loss_scale, 
                                        global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                                    writer.add_scalar('train/summary/scalar/all_fp16_gradients_finite_859', all_finite, 
                                        global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                            global_step += 1
                    # TODO: how to take named arguments 
                    else:
                        loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                     masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels,
                                     checkpoint_activations=args.checkpoint_activations)

                        if args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.

                        if args.gradient_accumulation_steps > 1:
                            if not args.allreduce_post_accumulation:
                                # this division was merged into predivision
                                loss = loss / args.gradient_accumulation_steps
                                divisor = 1.0

                        ##########################
                        # TODO: prevent loss.backward() call
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                     

                        if training_steps % args.gradient_accumulation_steps == 0:
                            global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)
                    average_loss += loss.item()
                    
                    if global_step >= args.max_steps:
                        last_num_steps = global_step % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        if is_main_process(args):
                            logger.info("Total Steps:{} Final Loss = {}".format(training_steps, average_loss.item()))
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if is_main_process(args):
                            writer.add_scalar('train/summary/total_loss', average_loss / (args.log_freq * divisor), 
                                    global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                            print("Step:{} Average Loss = {}".format(global_step, average_loss / (
                                        args.log_freq * divisor)))
                        average_loss = 0

                    if global_step >= args.max_steps or training_steps % (
                            args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
                        if is_main_process(args):
                            # Save a trained model
                            logger.info("** ** * Saving fine - tuned model ** ** * ")
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                            else:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
                            if args.do_train:
                                state = {'model': model_to_save.state_dict(),
                                         'files': [f_id] + files}
                                if not args.use_ort_trainer:
                                    state['optimizer'] = optimizer.state_dict()
                                    state['master params'] = list(amp.master_params(optimizer))
                                torch.save(state, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if global_step >= args.max_steps:

                            if is_main_process(args) and args.use_ort_trainer:
                                print('-----------------------save final onnx model-----------------------')
                                model_to_save.save_as_onnx('final_bert.onnx')
                            del train_dataloader
                            # thread.join()
                            return args
                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                train_dataloader, data_file = dataset_future.result(timeout=None)

            epoch += 1
    writer.close()

# I am getting RuntimeError: already started (lib/python/old_ptvsd/ptvsd/daemon.py", line 145) without following 
# https://github.com/microsoft/ptvsd/issues/1443
#import multiprocessing
#multiprocessing.set_start_method('spawn', True)

if __name__ == "__main__":
    now = time.time()
    args = main()
    if is_main_process(args):
        print("Total time taken {}".format(time.time() - now))

    print("main ends")
