from torch.utils.data import Dataset
from pathlib import Path
from vocoder import audio
import vocoder.hparams as hp
import numpy as np
import torch
import os
import tqdm
from rich.progress import track


class VocoderDataset(Dataset):
    def __init__(self, metadata_fpaths, mel_dirs, wav_dirs):
        '''
        Args:
            metadata_fpath: list
            mel_dir: list
            wav_dir: list
        '''
        gta_fpaths = []
        wav_fpaths = []
        alltime = 0

        self.samples_fpaths = []
        for i, metadata_fpath in enumerate(metadata_fpaths):
            print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dirs[i], wav_dirs[i]))
            with metadata_fpath.open("r") as metadata_file:
                metadata = [line.split("|") for line in metadata_file]

            # gta_fnames = [x[1] for x in metadata if int(x[4])]
            # gta_fpaths += [mel_dirs[i].joinpath(fname) for fname in gta_fnames]
            # wav_fnames = [x[0] for x in metadata if int(x[4])]
            # wav_fpaths += [wav_dirs[i].joinpath(fname) for fname in wav_fnames]
            for line in track(metadata):
                wav_path = os.path.join(wav_dirs[i], line[0].strip())
                pred_mel = os.path.join(mel_dirs[i], line[1].strip())
                mel_len = int(line[4])

                mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
                if not (mel_len - (mel_win + 2 * hp.voc_pad + 2) > 0):
                    print(wav_path)
                    continue
                alltime += int(line[3])/hp.sample_rate
                self.samples_fpaths.append((pred_mel,wav_path))

        print("Found {} samples, {} hours".format(len(self.samples_fpaths), alltime/3600))

    def __getitem__(self, index):
        mel_path, wav_path = self.samples_fpaths[index]
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
        mel = np.clip(mel, -1, 1)

        # Load the wav
        wav = np.load(wav_path)
        if hp.apply_preemphasis:
            wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)
        #
        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad = (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        assert len(wav) >= mel.shape[1] * hp.hop_length
        wav = wav[:mel.shape[1] * hp.hop_length]
        assert len(wav) % hp.hop_length == 0
        #
        # Quantize the wav
        if hp.voc_mode == 'RAW':
            if hp.mu_law:
                quant = audio.encode_mu_law(wav, mu=2 ** hp.bits)
            else:
                quant = audio.float_2_label(wav, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = audio.float_2_label(wav, bits=16)

        return mel.astype(np.float32), quant.astype(np.int64)

    def __len__(self):
        return len(self.samples_fpaths)


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] - 2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits
    # if hp.voc_mode == 'RAW':
    #     if hp.mu_law:
    #         x = audio.decode_mu_law(x.float(), 2 ** bits)
    #     else:
    x = audio.label_2_float(x.float(), bits=bits)
    if hp.voc_mode == 'MOL':
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels