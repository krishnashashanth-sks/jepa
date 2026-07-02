import torch

def update_ema_parameters(target_model, source_model, decay):
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)