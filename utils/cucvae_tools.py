import os
import sys
sys.path.append('./Griffin_lim')
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import librosa

from Griffin_lim.utils import audio
from Griffin_lim.hparams import hparams
matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            embeds,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            durations,
            mean_mels,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mean_mels = torch.from_numpy(mean_mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        # pitches = torch.from_numpy(pitches).float().to(device)
        # energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        embeds = torch.from_numpy(embeds).float().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            embeds,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            durations,
            mean_mels,
        )
    if len(data) == 7:
        (ids, raw_texts, speakers,  texts,embeds, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        embeds = torch.from_numpy(embeds).float().to(device)

        return (ids, raw_texts, speakers, texts, embeds, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/kl_loss", losses[2], step)
        logger.add_scalar("Loss/kl_loss_text", losses[3], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    mel_len = predictions[9+2][0].item() #9
    mel_target = targets[6+1][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[0][0, :mel_len].detach().transpose(0, 1)

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy()),
            (mel_target.cpu().numpy()),
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )
    if vocoder is not None:
        from .cucvae_model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    os.makedirs(os.path.join(path, 'hifigan'), exist_ok=True)
    os.makedirs(os.path.join(path, 'duration'), exist_ok=True)

    from .model import vocoder_infer

    mel_predictions = predictions[5].transpose(1, 2)
    lengths = predictions[9+2] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[10][i].item()
        duration = predictions[7][i, :src_len].detach().cpu().numpy()
        np.save(os.path.join(path, "duration/{}.npy".format(basename.replace(' ', '_'))), duration)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for d, wav, basename in zip(predictions[7], wav_predictions, basenames):
        hifi_save_path = os.path.join(path, "hifigan/{}.wav".format(basename.replace(' ', '_')))
        wavfile.write(hifi_save_path, sampling_rate, wav)

def plot_mel(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
