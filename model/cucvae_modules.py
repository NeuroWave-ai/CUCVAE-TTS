import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.cucvae_tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    """ VAE """

    def __init__(self, config):
        super(VAE, self).__init__()
        self.vae_dim = config["vae"]["dim"] # vae output dim
        vae_up_dim = config["vae"]["up_dim"] # the dim of upsample

        self.mu_fc = nn.Conv1d(in_channels=80, out_channels=self.vae_dim, kernel_size=1)
        self.logvar_fc = nn.Conv1d(in_channels=80, out_channels=self.vae_dim, kernel_size=1)
        self.up_fc = nn.Conv1d(in_channels=self.vae_dim, out_channels=vae_up_dim, kernel_size=1)

        self.text_mu_fc = nn.Conv1d(in_channels=256+1, out_channels=self.vae_dim, kernel_size=1)
        self.text_logvar_fc = nn.Conv1d(in_channels=256+1, out_channels=self.vae_dim, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)

    def reparameterize(self, mu, logvar, text_mu, text_logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())

        text_std = text_logvar.mul(0.5).exp_()
        text_eps = Variable(text_std.data.new(text_std.size()).normal_())
        text_prior = text_eps.mul(text_std).add_(text_mu)

        out = text_prior.mul(std).add_(mu)
        return out


    def forward(self, mel_mean, output, durations):
        durations = durations.unsqueeze(-1)
        output_in = torch.cat((durations, output), -1)
        output_ = output_in.permute(0, 2, 1)
        text_mu = self.text_mu_fc(output_)
        text_logvar = self.text_logvar_fc(output_)
        text_mu = text_mu.permute(0, 2, 1)
        text_logvar = text_logvar.permute(0, 2, 1)

        vae_in = mel_mean
        vae_in = vae_in.permute(0, 2, 1)
        vae_in = torch.nan_to_num(vae_in)
        mu = self.mu_fc(vae_in)
        logvar = self.logvar_fc(vae_in)

        mu = mu.permute(0, 2, 1)
        logvar = logvar.permute(0, 2, 1)

        z = self.reparameterize(mu, logvar, text_mu, text_logvar)
        # z = self.dropout(z)
        z = z.permute(0, 2, 1)
        z = self.up_fc(z)
        z = z.permute(0, 2, 1)
        return z, mu, logvar, text_mu, text_logvar

    def inference(self, input_shape, output, durations):
        durations = durations.unsqueeze(-1)
        output_in = torch.cat((durations, output), -1)
        mu = torch.zeros([input_shape[0], input_shape[1], self.vae_dim]).cuda()
        logvar = torch.zeros([input_shape[0], input_shape[1], self.vae_dim]).cuda()

        output_ = output_in.permute(0, 2, 1)
        text_mu = self.text_mu_fc(output_)
        text_logvar = self.text_logvar_fc(output_)
        text_mu = text_mu.permute(0, 2, 1)
        text_logvar = text_logvar.permute(0, 2, 1)

        z = self.reparameterize(mu, logvar, text_mu, text_logvar)

        z = z.permute(0, 2, 1)
        z = self.up_fc(z)
        z = z.permute(0, 2, 1)
        return z, mu, logvar, text_mu, text_logvar

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.VAE = VAE(model_config)

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        duration_target=None,
        mean_mels=None,
        d_control=1.0,
    ):

        x_vae = x.clone()
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)
        if mean_mels==None:
            print('inference-ing')
            vae_embedding, mu, logvar, text_mu, text_logvar = self.VAE.inference(x_vae.size(), x_vae, duration_rounded)
        else:
            vae_embedding, mu, logvar, text_mu, text_logvar = self.VAE(mean_mels, x_vae, duration_rounded)


        vae_embedding, _ = self.length_regulator(vae_embedding, duration_rounded, max_len)
        x = x + vae_embedding
        return (
            x,
            mu,
            logvar,
            text_mu,
            text_logvar,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

