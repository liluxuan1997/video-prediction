import torch

def cross_entropy(outputs, targets):
    # [batch, seq_length, channel, width, height]
    return torch.sum(-targets*torch.log(outputs)-(1-targets)*torch.log(1-outputs))

def psnr(outputs, targets):
    # [batch, seq_length, channel, width, height]
    num_pixels = outputs.shape[2] * outputs.shape[3] * outputs.shape[4]
    batch_size = outputs.shape[0]
    seq_length = outputs.shape[1]
    psnr = torch.zeros((outputs.shape[0],outputs.shape[1]))
    for i in range(batch_size):
        for j in range(seq_length):
            mse = torch.mean((outputs[i,j,:,:,:] - targets[i,j,:,:,:])**2)
            psnr[i,j] = 20 * torch.log10(torch.max(outputs[i,j,:,:,:])) - 10 * torch.log10(mse)
    return torch.sum(psnr)

