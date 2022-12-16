import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

from configs import get_config
import audio
import utils
import dataset
import text
import model as M
import waveglow_local

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = get_config()

def get_DNN(num):
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(str(num))['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    test1 = "Accept the things to which fate binds you, and love the people with whom fate brings you together, but do so with all your heart."
    test2 = "We suffer more often in imagination than in reality."
    test3 = "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space."
    data_list = list()
    data_list.append(text.text_to_sequence(test1, cfg.text_cleaners))
    data_list.append(text.text_to_sequence(test2, cfg.text_cleaners))
    data_list.append(text.text_to_sequence(test3, cfg.text_cleaners))
    return data_list


if __name__ == "__main__":
    # Test
    WaveGlow = utils.get_WaveGlow()
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='')
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    print("use griffin-lim and waveglow")
    model = get_DNN(args.model_name)
    data_list = get_data()
    for i, phn in enumerate(data_list):
        mel, mel_cuda = synthesis(model, phn, args.alpha)
        if not os.path.exists("results"):
            os.mkdir("results")
        audio.tools.inv_mel_spec(
            mel, str(args.model_name).replace(".pth.tar", f"_{i}.wav"))
        waveglow_local.inference.inference(
            mel_cuda, WaveGlow,
            str(args.model_name).replace(".pth.tar", f"_{i}_waveglow.wav"))
        print("Done", i + 1)

    s_t = time.perf_counter()
    for i in range(100):
        for _, phn in enumerate(data_list):
            _, _, = synthesis(model, phn, args.alpha)
        print(i)
    e_t = time.perf_counter()
    print((e_t - s_t) / 100.)
