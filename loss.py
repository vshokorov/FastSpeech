import torch
import torch.nn as nn
from configs import get_config

cfg = get_config()

class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        scale = (cfg.mel_loss_scale + cfg.mel_postnet_loss_scale + cfg.duration_loss_scale)
        self.mel_loss_scale         = cfg.mel_loss_scale / scale
        self.mel_postnet_loss_scale = cfg.mel_postnet_loss_scale / scale
        self.duration_loss_scale    = cfg.duration_loss_scale / scale


    def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target):
        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        duration_predictor_target.requires_grad = False
        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        total_loss = (mel_loss * self.mel_loss_scale + 
                      mel_postnet_loss * self.mel_postnet_loss_scale + 
                      duration_predictor_loss * self.duration_loss_scale)

        return total_loss, mel_loss, mel_postnet_loss, duration_predictor_loss
