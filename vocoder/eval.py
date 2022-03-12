from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.display import stream, simple_table
from vocoder.gen_wavernn import gen_testset
from torch.utils.data import DataLoader
from pathlib import Path
from torch import optim
import torch.nn.functional as F
import vocoder.hparams as hp
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from pathlib import Path


def eval(run_id: str, models_dir: Path, ground_truth: bool, force_restart: bool):
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    ).cuda()

    print("Is using CUDA:{}".format(torch.cuda.is_available()))
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

    # Load the weights
    model_dir = models_dir.joinpath(run_id)
    weights_fpath = model_dir.joinpath(run_id + ".pt")

    print("\nLoading weights at %s" % weights_fpath)
    model.load(weights_fpath, optimizer)
    print("WaveRNN weights loaded from step %d" % model.step)
    # Initialize the dataset
    syn_dirs = hp.syn_dirs
    voc_dirs = hp.voc_dirs

    metadata_fpath = []
    metadata_fpath_test = []
    mel_dir = []
    wav_dir = []
    for i,syn_dir in enumerate(syn_dirs):
        metadata_fpath.append(syn_dir.joinpath("train.txt") if ground_truth else \
            voc_dirs[i].joinpath("synthesized.txt"))
        metadata_fpath_test.append(syn_dir.joinpath("test.txt") if ground_truth else \
            voc_dirs[i].joinpath("synthesized_test.txt"))
        mel_dir.append(syn_dir.joinpath("mels") if ground_truth else voc_dirs[i].joinpath("mels_gta"))
        wav_dir.append(syn_dir.joinpath("audio"))
    dataset_test = VocoderDataset(metadata_fpath_test, mel_dir, wav_dir)


    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    start = time.time()
    running_loss = 0.

    test_loader = DataLoader(dataset_test,
                             collate_fn=collate_vocoder,
                             batch_size=hp.voc_batch_size,
                             num_workers=8,
                             shuffle=True,
                             pin_memory=True)
    start = time.time()
    running_loss = 0.
    with torch.no_grad():
        for i, (x, y, m) in enumerate(test_loader, 1):
            x, m, y = x.cuda(), m.cuda(), y.cuda()

            # Forward pass
            y_hat = model(x, m)
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)

            # Backward pass
            loss = loss_func(y_hat, y)

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss_test = running_loss / i

            
            msg = f"| ({i}/{len(test_loader)}) | " \
                  f"Loss: {avg_loss_test:.4f} | {speed:.1f} " \
                  f"steps/s | "
            stream(msg)