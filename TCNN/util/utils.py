import importlib
import json
import math
import os
import random
import time

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False) 
    for j, seq_len in enumerate(nframes):
        loss_mask.data[j, :, 0:seq_len] += 1.0
    return loss_mask

def sliceframe(in_sig):
    frame_size = 320
    frame_shift = 160
    sig_len = in_sig.shape[-1]
    nframes = (sig_len // frame_shift) 
    a = np.zeros(list(in_sig.shape[:-1]) + [nframes, frame_size])
    start = 0
    end = start + frame_size
    k=0
    for i in range(nframes):
        if end < sig_len:
            a[..., i, :] = in_sig[..., start:end]
            k += 1
        else:
            tail_size = sig_len - start
            a[..., i, :tail_size]=in_sig[..., start:]
                
        start = start + frame_shift
        end = start + frame_size
    return a   

def OverlapAndAdd(inputs,frame_shift):
        nframes = inputs.shape[-2]
        print(nframes)
        frame_size = inputs.shape[-1]
        print(frame_size)
        frame_step = frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones



class ExecutionTime:

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time


def find_aligned_wav_files(dir_a, dir_b, limit=0, offset=0):
   

    if limit == 0:
       
        limit = None

    wav_paths_in_dir_a = librosa.util.find_files(dir_a, ext="wav", limit=limit, offset=offset)
    wav_paths_in_dir_b = librosa.util.find_files(dir_b, ext="wav", limit=limit, offset=offset)

    length = len(wav_paths_in_dir_a)

  
    assert len(wav_paths_in_dir_a) == len(wav_paths_in_dir_b) > 0, f"{dir_a}  {dir_b} "

 
    for wav_path_a, wav_path_b in zip(wav_paths_in_dir_a, wav_paths_in_dir_b):
        assert os.path.basename(wav_path_a) == os.path.basename(wav_path_b), \
            f"{wav_path_a}"

    return wav_paths_in_dir_a, wav_paths_in_dir_b, length


def set_requires_grad(nets, requires_grad=False):


    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
   
    assert len(data_a) == len(data_b), "length should be equal"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]


def sample_dataset_aligned(dataset_A, dataset_B, n_frames=128):

    data_A_idx = np.arange(len(dataset_A))
    data_B_idx = np.arange(len(dataset_B))

    sampling_dataset_A = list()
    sampling_dataset_B = list()

    for idx_A, idx_B in zip(data_A_idx, data_B_idx):
    
        data_A = dataset_A[idx_A]
        data_B = dataset_B[idx_B]

  
        frames_A_total = data_A.shape[1]
        frames_B_total = data_B.shape[1]
        assert frames_A_total == frames_B_total, "A and B equal {}.".format(idx_A)

        assert frames_A_total >= n_frames
        start = np.random.randint(frames_A_total - n_frames + 1)
        end = start + n_frames
        sampling_dataset_A.append(data_A[:, start: end])
        sampling_dataset_B.append(data_B[:, start: end])

    sampling_dataset_A = np.array(sampling_dataset_A)
    sampling_dataset_B = np.array(sampling_dataset_B)

    return sampling_dataset_A, sampling_dataset_B


def calculate_l_out(l_in, kernel_size, stride, dilation=1, padding=0):
    # https://pytorch.org/docs/stable/nn.html#conv1d
    return math.floor(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)


def calculate_same_padding(l_in, kernel_size, stride, dilation=1):
    # https://pytorch.org/docs/stable/nn.html#conv1d
    return math.ceil(((l_in - 1) * stride + 1 + dilation * (kernel_size - 1) - l_in) / 2)


def initialize_config_in_single_module(module_cfg, module):
  
    return getattr(module, module_cfg["type"])(**module_cfg["args"])


def initialize_config(module_cfg):
   
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def write_json(content, path):
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2, sort_keys=False)


def apply_mean_std(y):
    return (y - np.mean(y)) / np.std(y)


def cal_lps(y, pad=0):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    mag = np.abs(D)
    lps = np.log(np.power(mag, 2))
    if (pad != 0):
        lps = np.concatenate((np.zeros((257, pad)), lps, np.zeros((257, pad))), axis=1)
    return lps


def mag(y):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    return np.abs(D)


def input_normalization(m):
    mean = np.mean(m, axis=0)
    std_var = np.std(m, axis=0)
    return (m - mean) / std_var


def unfold_spectrum(spec, n_pad=3):
    
    left_pad_spec = np.repeat(spec[:, 0].reshape(-1, 1), n_pad, axis=1)  # (257, 3)
    right_pad_spec = np.repeat(spec[:, -1].reshape(-1, 1), n_pad, axis=1)  # (257, 3)
    assert left_pad_spec.shape[-1] == right_pad_spec.shape[-1] == n_pad
    spec = np.concatenate([left_pad_spec, spec, right_pad_spec], axis=1).T  # (120, 257)
    spec = torch.Tensor(spec)

   
    spec_list = spec.unfold(0, 2 * n_pad + 1, 1)  # [tensor(257, 7), tensor(257, 7), ...], len = 114
    spec = torch.cat(tuple(spec_list), dim=1).numpy()  # (257, 798)

    return spec


def lps_to_mag(lps):
    return np.power(np.exp(lps), 1 / 2)


def rebuild_waveform(mag, noisy_phase):
    return librosa.istft(mag * noisy_phase, hop_length=256, win_length=512, window='hamming')


def phase(y):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    _, phase = librosa.magphase(D)
    return phase


def add_noise_for_waveform(s, n, db):
    alpha = np.sqrt(
        np.sum(s ** 2) / (np.sum(n ** 2) * 10 ** (db / 10))
    )
    mix = s + alpha * n
    return mix


def synthesis_noisy_y(clean_y, noise_y, snr):


    assert len(clean_y) > 0 and len(noise_y) > 0, f"The length of the noise file is {len(noise_y)}, and the length of the clean file is {len(clean_y)}."
    assert type(snr) == str, "Specify the snr of the string type."

    if len(clean_y) >= len(noise_y):
      
        pad_factor = len(clean_y) // len(noise_y)  
        padded_noise_y = noise_y
        for i in range(pad_factor):
            padded_noise_y = np.concatenate((padded_noise_y, noise_y))
        noise_y = padded_noise_y

    # Randomly crop noise segment, the length of the noise segment is equal to the length of the clean file.
    s = random.randint(0, len(noise_y) - len(clean_y) - 1)
    e = s + len(clean_y)
    noise_y = noise_y[s:e]
    assert len(noise_y) == len(clean_y), f"The length of the noise file is {len(noise_y)}, and the length of the clean file is {len(clean_y)}."

    noisy_y = add_noise_for_waveform(clean_y, noise_y, int(snr))
    return clean_y, noise_y, noisy_y

def prepare_device(n_gpu: int, cudnn_deterministic=False):
    """Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0")

    return device


def prepare_empty_dir(dirs, resume=False):
  
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def pad_to_longest_in_one_batch(batch):
    
    noisy_mag_list = []
    clean_mag_list = []
    n_frames_list = []

    for noisy_mag, clean_mag, n_frames in batch:
        noisy_mag_list.append(torch.tensor(noisy_mag)) 
        clean_mag_list.append(torch.tensor(clean_mag))
        n_frames_list.append(n_frames)

    noisy_mag_one_batch = pad_sequence(noisy_mag_list)  
    clean_mag_one_batch = pad_sequence(clean_mag_list)


    noisy_mag_one_batch = noisy_mag_one_batch.permute(1, 0, 2)  
    clean_mag_one_batch = clean_mag_one_batch.permute(1, 0, 2)

    noisy_mag_one_batch = noisy_mag_one_batch[:,None,:,:]
    clean_mag_one_batch = clean_mag_one_batch[:,None,:,:]


    return noisy_mag_one_batch, clean_mag_one_batch, n_frames_list
