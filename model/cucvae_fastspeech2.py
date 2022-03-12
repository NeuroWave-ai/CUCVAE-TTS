import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .cucvae_modules import VarianceAdaptor, VAE
from utils.cucvae_tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.pre_multihead_attn = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)
        self.multihead_attn = nn.MultiheadAttention(256, 16)
        self.post_multihead_attn = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        embeds,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        d_targets=None,
        mean_mels=None,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        text_embedding = self.encoder(texts, src_masks)
        output = text_embedding.clone() # deep copy
        if self.speaker_emb is not None:
            speaker_embedding = self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
            output = output + speaker_embedding

        embeds = embeds.permute(0, 2, 1)
        embeds = self.pre_multihead_attn(embeds)
        embeds = embeds.permute(0, 2, 1)

        embeds_ = embeds.permute(1, 0, 2)
        output_ = output.permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(query=output_, key=embeds_, value=embeds_)
        attn_output = attn_output.permute(1, 0, 2)

        attn_output = torch.cat([attn_output, output],-1)
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = self.post_multihead_attn(attn_output)
        cu_embedding = attn_output.permute(0, 2, 1)
        output = output + cu_embedding
        

        (
            output,
            mu,
            logvar,
            text_mu,
            text_logvar,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            d_targets,
            mean_mels,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            mu,
            logvar,
            text_mu,
            text_logvar,
            postnet_output, #5
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens, # 9
        )
