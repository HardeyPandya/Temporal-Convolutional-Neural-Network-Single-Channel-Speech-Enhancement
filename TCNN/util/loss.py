import torch
from torch.nn.utils.rnn import pad_sequence

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

def bce_loss():
    return torch.nn.BCEWithLogitsLoss()

def mse_loss_for_variable_length_data():
    def loss_function(outputs, labels, loss_mask, nframes):
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        loss = torch.sum((masked_outputs - masked_labels)**2.0) / torch.sum(loss_mask)
        return loss
      
    return loss_function
      
