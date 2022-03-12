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
import torch.distributed as dist
import torch.nn as nn
gpus = [0, 1]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

def train(run_id: str, models_dir: Path, ground_truth: bool, force_restart: bool):
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
    )
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    print('dataset : {}'.format(hp.syn_dirs))
    print("Is using CUDA:{}".format(torch.cuda.is_available()))
    print('Is apply_preemphasis:{}'.format(hp.apply_preemphasis))
    print('Is apply_mulaw:{}'.format(hp.mu_law))
    print('GRU dims:{}'.format(hp.voc_rnn_dims))
    print('is width GRU:{}'.format(hp.width_rnn))
    print('if width gru the hidden size of gru is 512*2, and add a projection layer following every gru.')
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr
    loss_func = F.cross_entropy if model.module.mode == "RAW" else discretized_mix_logistic_loss

    # Load the weights
    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)
    wavs_dir = model_dir.joinpath('wavs')
    wavs_dir.mkdir(exist_ok=True)
    backup_dir = model_dir.joinpath('backup')
    backup_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(model_dir.joinpath('logs'))
    weights_fpath = model_dir.joinpath(run_id + ".pt")
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.module.save(weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.module.load(weights_fpath, optimizer)
        start_epoch = 0
        print("WaveRNN weights loaded from step %d" % model.module.step)
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
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    dataset_test = VocoderDataset(metadata_fpath_test, mel_dir, wav_dir)

    gen_test_loader = DataLoader(dataset_test,
                                 batch_size=1,
                                 shuffle=True,
                                 pin_memory=True)
    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, hp.milestones, gamma=hp.MultiStepLR_gamma, last_epoch=-1)
    for epoch in range(start_epoch, 2000):
        print("epoch:{} start training before collect data".format(epoch))
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_vocoder,
                                 batch_size=hp.voc_batch_size,
                                 num_workers=8,
                                 shuffle=True,
                                 pin_memory=True)

        print("epoch:{} start training after collect data".format(epoch))
        start = time.time()
        running_loss = 0.
        
        for i, (x, y, m) in enumerate(data_loader, 1):
            x, m, y = x.cuda(), m.cuda(), y.cuda()

            # Forward pass
            y_hat = model(x, m)
            if model.module.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.module.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)

            # Backward pass
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if np.isnan(grad_norm.cpu()):
                    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.module.get_step()
            k = step // 1000

            if step % 50 == 0:
                msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                      f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                      f"steps/s | Step: {k}k | "
                stream(msg)

        if epoch%10==0 or epoch==1:

            gen_testset(model, gen_test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                        hp.voc_target, hp.voc_overlap, model_dir,epoch)

            print("epoch:{} start testing before collect data".format(epoch))
            test_loader = DataLoader(dataset_test,
                                     collate_fn=collate_vocoder,
                                     batch_size=hp.voc_batch_size,
                                     num_workers=8,
                                     shuffle=True,
                                     pin_memory=True)
            print("epoch:{} start test after collect data".format(epoch))
            start = time.time()
            running_loss = 0.
            with torch.no_grad():
                for i, (x, y, m) in enumerate(test_loader, 1):
                    x, m, y = x.cuda(), m.cuda(), y.cuda()

                    # Forward pass
                    y_hat = model(x, m)
                    if model.module.mode == 'RAW':
                        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                    elif model.module.mode == 'MOL':
                        y = y.float()
                    y = y.unsqueeze(-1)

                    # Backward pass
                    loss = loss_func(y_hat, y)

                    running_loss += loss.item()
                    speed = i / (time.time() - start)
                    avg_loss_test = running_loss / i

                    if step % 50 == 0:
                        msg = f"| Epoch: {epoch} ({i}/{len(test_loader)}) | " \
                              f"Loss: {avg_loss_test:.4f} | {speed:.1f} " \
                              f"steps/s | Step: {step} | "
                        stream(msg)

            writer.add_scalar('Loss_val', avg_loss_test, epoch)
        writer.add_scalar('Loss_train', avg_loss, epoch)
        model.module.save(weights_fpath, optimizer)
        model.module.checkpoint(backup_dir, optimizer, epoch)
        scheduler.step()
        print("")
