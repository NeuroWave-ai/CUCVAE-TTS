import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.beta = model_config['vae']['beta']

    def KLD_loss(self, mu, logvar):
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            duration_targets,
            _,
        ) = inputs[6+1:]
        (
            mel_predictions,
            mu,
            logvar,
            text_mu,
            text_logvar,
            postnet_mel_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False


        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        kl_loss = self.KLD_loss(mu,logvar)
        kl_loss_text = self.KLD_loss(text_mu,text_logvar)
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + self.beta*(kl_loss + kl_loss_text)
        )

        return (
            total_loss,
            mel_loss,
            kl_loss,
            kl_loss_text,
            postnet_mel_loss,
            duration_loss,
        )
