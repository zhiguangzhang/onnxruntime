import os
import random
from run_pretraining import pretraining_dataset, create_pretraining_dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm, trange
from utils import is_main_process as orginal_is_main_process
from concurrent.futures import ProcessPoolExecutor

from onnxruntime.capi.ort_trainer import LossScaler

def is_main_process(args):
    if hasattr(args, 'world_size'):
        world_rank = get_world_rank() if 'RANK' in os.environ.keys() else args.local_rank
        return world_rank == 0
    else:
        return original_is_main_process()

from optimization import SCHEDULES
def get_lr(args, training_steps, schedule='warmup_poly'):
    if args.max_steps == -1:
        return args.learning_rate

    schedule_fct = SCHEDULES[schedule]
    return args.learning_rate * schedule_fct(training_steps / args.max_steps, args.warmup_proportion)

def tb_tag(args, base_tag):
    tag = base_tag
    if args.phase2:
        tag += "phase2_"
    else:
        tag += "phase1_"
    
    if args.fp16:
        tag += "mixed_precision_"
    else:
        tag += "full_precision_"
    
    tag += ("batch-" + str(args.train_batch_size) + "max_seq_length-" + str(args.max_seq_length) +
        "gradient_accumulation_steps-" + str(args.gradient_accumulation_steps))
    return tag


def train_with_ort_trainer(args, model, checkpoint, global_step, device, logger):
    loss_tag = tb_tag(args, "train/fully_optimized_summary_4/total_loss_")
    lr_tag = tb_tag(args, "train/fully_optimized_summary_4/lr_")
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

        if args.use_ort_trainer_mixed_precision:
            loss_scaler = LossScaler(model.loss_scale_input_name, True)

        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            if not args.resume_from_checkpoint or epoch > 0 or args.phase2:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f))
                        # and 'training' in f
                        ]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                args.resume_from_checkpoint = False
                num_files = len(files)


            shared_file_list = {}

            ############
            if torch.distributed.is_initialized():
                # launched with MPI torch.distributed.launch:
                world_size = torch.distributed.get_world_size()
            elif hasattr(args, 'world_size'):
                # launched with MPI
                world_size = args.world_size
            else:
                world_size = 1

            if world_size > num_files:
                remainder = world_size % num_files
                data_file = files[(f_start_id*world_size+args.local_rank + remainder*f_start_id)%num_files]
            else:
                data_file = files[(f_start_id*world_size+args.local_rank)%num_files]
            ############
            # # # if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
            # # #     remainder = torch.distributed.get_world_size() % num_files
            # # #     data_file = files[(f_start_id*torch.distributed.get_world_size()+torch.distributed.get_rank() + remainder*f_start_id)%num_files]
            # # # elif torch.distributed.is_initialized():
            # # #     # TODO: This following line fails if not torch.distributed.is_initialized().
            # # #     data_file = files[(f_start_id*torch.distributed.get_world_size()+torch.distributed.get_rank())%num_files]
            # # # else:
            # # #     data_file = files[f_start_id % num_files]
            ############

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size * args.n_gpu, num_workers=4,
                                          pin_memory=True)
            # shared_file_list["0"] = (train_dataloader, data_file)

            overflow_buf = None
            if args.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])

            for f_id in range(f_start_id + 1 , len(files)):
                
                if world_size > num_files:
                    data_file = files[(f_id*world_size+args.local_rank + remainder*f_id)%num_files]
                elif world_size > 1:
                    data_file = files[(f_id*world_size+args.local_rank)%num_files]
                else:
                    data_file = files[f_id%num_files]
                # # # # # if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                # # # # #     data_file = files[(f_id*torch.distributed.get_world_size()+torch.distributed.get_rank() + remainder*f_id)%num_files]
                # # # # # elif torch.distributed.is_initialized():
                # # # # #     data_file = files[(f_id*torch.distributed.get_world_size()+torch.distributed.get_rank())%num_files]
                # # # # # else:
                # # # # #     data_file = files[f_id%num_files]

                logger.info("file no %s file %s" % (f_id, previous_file))

                previous_file = data_file

                dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args)

                train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process(args) else train_dataloader
                # train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader
                for step, batch in enumerate(train_iter):

                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

                    lr = get_lr(args, global_step, args.schedule)
                    learning_rate = torch.tensor([lr])

                    # print("type(model):", type(model))
                    # print("model.__dir__():", model.__dir__())
                    # import pdb; pdb.set_trace()

                    if args.use_ort_trainer_mixed_precision:
                        loss_scale = torch.tensor(loss_scaler.loss_scale_)
                        loss = model.train_step((input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate, loss_scale))
                        if isinstance(loss, (list, tuple)):
                            assert len(loss) == 2
                            loss, all_finite = loss
                            loss_scaler.update_loss_scale(all_finite.item())
                    else:
                        loss = model.train_step((input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate))

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    divisor = args.gradient_accumulation_steps
                    if args.gradient_accumulation_steps > 1:
                        if not args.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / args.gradient_accumulation_steps
                            divisor = 1.0

                    ##########################
                    # TODO: prevent loss.backward() call
                    if args.use_ort is False:
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                     
                    average_loss += loss.item()

                    if training_steps % args.gradient_accumulation_steps == 0:
                        global_step += 1

                    if global_step >= args.max_steps:
                        last_num_steps = global_step % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            # # # # # average_loss /= torch.distributed.get_world_size()
                            average_loss /= world_size()
                            torch.distributed.all_reduce(average_loss)
                        if is_main_process(args):
                            logger.info("Total Steps:{} Final Loss = {}".format(training_steps, average_loss.item()))
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if is_main_process(args):
                            writer.add_scalar(loss_tag, average_loss / (args.log_freq * divisor), 
                                    global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                            writer.add_scalar(lr_tag, lr, 
                                    global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)                            
                            print("Step:{} Average Loss = {} Step Loss = {} LR {}".format(global_step, average_loss / (
                                        args.log_freq * divisor),
                                                                                            loss.item() * args.gradient_accumulation_steps / divisor,
                                                                                            lr))
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
                                # TODO: do checkpoint with ORTTrainer.
                                # this code currently works with limitations - it misses optimizer state. 
                                torch.save({'model': model_to_save.state_dict(),
                                            # 'optimizer': optimizer.state_dict(),
                                            # 'master params': list(amp.master_params(optimizer)),
                                            'files': [f_id] + files}, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if global_step >= args.max_steps:
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
