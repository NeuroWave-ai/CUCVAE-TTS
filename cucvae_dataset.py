import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.cucvae_tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.embed_dir = train_config["embed_dir"]
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        # pitch_path = os.path.join(
        #     self.preprocessed_path,
        #     "pitch",
        #     "{}-pitch-{}.npy".format(speaker, basename),
        # )
        # pitch = np.load(pitch_path)
        # energy_path = os.path.join(
        #     self.preprocessed_path,
        #     "energy",
        #     "{}-energy-{}.npy".format(speaker, basename),
        # )
        # energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        embeds_path = os.path.join(
            self.preprocessed_path,
            self.embed_dir,
            "{}-embeds-{}.npy".format(speaker, basename),
        )
        embeds = np.load(embeds_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "duration": duration,
            "embeds": embeds
        }
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def mean_mels(self, mels, durations):
        assert len(mels) == len(durations)
        mean_mels = []
        for i in range(len(durations)):
            mel = mels[i]
            duration = durations[i]
            mean_mel = np.zeros([len(duration), mel.shape[1]])
            start = 0
            for j, d in enumerate(durations[i]):
                mean = np.mean(mel[start:start+d], axis=0)
                mean_mel[j] = mean
                start += d
            mean_mels.append(mean_mel)
        return mean_mels

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        embeds = np.array([data[idx]["embeds"] for idx in idxs])
        mean_mels = self.mean_mels(mels, durations)

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        durations = pad_1D(durations)
        mean_mels = pad_2D(mean_mels)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            embeds,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            durations,
            mean_mels,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, embed_dir):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.embed_dir = embed_dir
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        embeds_path = os.path.join(
            self.preprocessed_path,
            self.embed_dir,
            "{}-embeds-{}.npy".format(speaker, basename),
        )
        embeds = np.load(embeds_path)
        return (basename, speaker_id, phone, raw_text, embeds)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        embeds = np.array([d[4] for d in data])
        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts,embeds, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LibriTTS/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LibriTTS/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            print(len(batch))
            exit(0)
            #to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            print(batch)
            # to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )