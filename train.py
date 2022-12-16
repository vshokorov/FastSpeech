import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

from model import FastSpeech
from loss import DNNLoss
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from optimizer import ScheduledOptim
from configs import get_config
import utils
import wandb


def main(cfg):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    print("Use FastSpeech")
    model = nn.DataParallel(FastSpeech()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)
    # Get buffer
    print("Load data to buffer")
    buffer = get_data_to_buffer()

    for n, p in model.named_parameters():
        if cfg.freeze_encoder and n.startswith('module.encoder'):
            p.requires_grad = False
            print('Freeze', n)
        elif cfg.freeze_length_regulator and n.startswith('module.length_regulator'):
            p.requires_grad = False
            print('Freeze', n)
        elif cfg.freeze_decoder and n.startswith('module.decoder'):
            p.requires_grad = False
            print('Freeze', n)
        elif cfg.freeze_mel_linear and n.startswith('module.mel_linear'):
            p.requires_grad = False
            print('Freeze', n)
        elif cfg.freeze_postnet and n.startswith('module.postnet'):
            p.requires_grad = False
            print('Freeze', n)

    # Optimizer and loss
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 betas=(0.9, 0.98),
                                 eps=1e-9)
    # scheduled_optim = ScheduledOptim(optimizer,
    #                                  cfg.decoder_dim,
    #                                  cfg.n_warm_up_step,
    #                                  cfg.restore_step)
    fastspeech_loss = DNNLoss().to(device)
    print("Defined Optimizer and Loss Function.")
    
    wandb.init(
        project="FastSpeechStudy", 
        entity="vetrov_disciples",
        name=cfg.run_name,
        # mode='offline',
    )
    wandb.config.update(cfg)
    wandb.watch(model)

    # Load checkpoint if exists
    checkpoint = {}
    try:
        checkpoint = torch.load(os.path.join(
            cfg.init_weights, 'checkpoint_%d.pth.tar' % cfg.restore_step))
    except:
        print("\n---No checkpoint. Start New Training---\n")
    
    try:
        model.load_state_dict(checkpoint['model'])
        print("\n---Model Restored at Step %d---\n" % cfg.restore_step)
    except:
        print("\n---Error while loading model. Init new weights---\n")

    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Optimizer Restored at Step %d---\n" % cfg.restore_step)
    except:
        print("\n---Error while loading optimizer. Init new one---\n")

    if not os.path.exists(cfg.checkpoint_path):
        os.mkdir(cfg.checkpoint_path)

    # Init logger
    if not os.path.exists(cfg.logger_path):
        os.mkdir(cfg.logger_path)

    # Get dataset
    dataset = BufferDataset(buffer)

    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=cfg.batch_expand_size * cfg.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=4)
    total_step = cfg.epochs * len(training_loader) * cfg.batch_expand_size

    scheduled_optim = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.learning_rate,
        total_step
    )
    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()

    for epoch in range(cfg.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i * cfg.batch_expand_size + j + cfg.restore_step + \
                    epoch * len(training_loader) * cfg.batch_expand_size + 1

                # Init
                optimizer.zero_grad()

                # Get Data
                character = db["text"].long().to(device)
                mel_target = db["mel_target"].float().to(device)
                duration = db["duration"].int().to(device)
                mel_pos = db["mel_pos"].long().to(device)
                src_pos = db["src_pos"].long().to(device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, mel_postnet_output, duration_predictor_output = model(character,
                                                                                  src_pos,
                                                                                  mel_pos=mel_pos,
                                                                                  mel_max_length=max_mel_len,
                                                                                  length_target=duration)

                # Cal Loss
                total_loss, mel_loss, mel_postnet_loss, duration_loss = fastspeech_loss(mel_output,
                                                                                        mel_postnet_output,
                                                                                        duration_predictor_output,
                                                                                        mel_target,
                                                                                        duration)

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = duration_loss.item()

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                clip_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_thresh)
                
                wandb.log({
                    "step": current_step,
                    "total_loss": t_l,
                    "mel_loss": m_l,
                    "mel_postnet_loss": m_p_l,
                    "duration_loss": d_l,
                    # "lr": scheduled_optim.get_learning_rate(),
                    "lr": scheduled_optim.get_last_lr()[0],
                    "clip_norm": clip_norm
                })

                # Update weights
                # if cfg.frozen_learning_rate:
                #     scheduled_optim.step_and_update_lr_frozen(
                #         cfg.learning_rate)
                # else:
                #     scheduled_optim.step_and_update_lr()
                optimizer.step()
                scheduled_optim.step()

                # Print
                if current_step % cfg.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, cfg.epochs, current_step, total_step)
                    str2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f};".format(
                        m_l, m_p_l, d_l)
                    str3 = "Current Learning Rate is {:.6f}.".format(
                        # scheduled_optim.get_learning_rate())
                        scheduled_optim.get_last_lr()[0])
                    str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    print(str4)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write("\n")

                if current_step % cfg.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(cfg.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == cfg.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument('--restore_step', type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    if not args.restore_step is None:
        cfg.restore_step = args.restore_step

    main(cfg)
