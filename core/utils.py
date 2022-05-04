import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm #as sp_norm
import numpy as np
import time
import os


def fix_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.exp_name, "progress.txt")
        with open(self.file_name, "a") as log_file:
            log_file.write('--- Started training --- \n')

    def __call__(self, epoch):
        if epoch != 0:
            avg = (time.time() - self.prev_time) / (epoch - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = epoch

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d], avg time:%.3f per epoch \n' % (epoch, self.num_epochs, avg))
        print('[epoch %d/%d], avg time:%.3f per epoch' % (epoch, self.num_epochs, avg))
        return avg


def update_EMA(netEMA, netG, EMA_decay):
    with torch.no_grad():
        for key in netG.state_dict():
            netEMA.state_dict()[key].data.copy_(
                netEMA.state_dict()[key].data * EMA_decay +
                netG.state_dict()[key].data * (1 - EMA_decay)
            )
    return netEMA


def preprocess_real(batch, num_blocks_ll, device):
    # --- Put everything on GPU if needed --- #
    for item in batch:
        batch[item] = batch[item].to(device)
    #print(batch["images"].shape)
    # --- Create downsampled versions of real images for MSG --- #
    ans = list()
    image = batch["images"]
    #print(image)
    #print(image.shape)#torch.Size([5, 3, 80, 80, 80])
    ans.append(image)
    #print(ans)
    #print(num_blocks_ll)
    for i in range(num_blocks_ll-1):
        #image = F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)#利用插值方法，对输入的张量数组进行下采样操作，缩小一半>
        image = F.interpolate(image, scale_factor=0.5, mode="trilinear", align_corners=False, recompute_scale_factor=False)


        #image = F.interpolate(image, size=(40,40,40), mode="Trilinear", align_corners=True)

        #(input, output_size, align_corners, scale_factors
        #print(image)
        ans.append(image)          #imag是需要进行采样处理的数组
       # print(ans)
    batch["images"] = list(reversed(ans))
    return batch


def sample_noise(noise_dim, batch_size):
    #return torch.randn(batch_size, noise_dim, 1, 1)
    return torch.randn(batch_size, noise_dim, 1, 1, 1) #(5,64,1,1)


def to_rgb(in_channels):
    #return sp_norm(nn.Conv3d(in_channels, 3, (3, 3), padding=(1, 1), bias=True)) #第二个参数是输出通道数目；（3，3）是卷积核大小；padding=(1,1)表示在上、下、左、右四个方向各补一行0
    return torch.nn.utils.spectral_norm(nn.Conv3d(in_channels, 3, (3, 3, 3), padding=(1, 1, 1), bias=True))

def get_norm_by_name(norm_name, out_channel):
    if norm_name == "batch":
        return nn.BatchNorm3d(out_channel)
    if norm_name == "instance":
        return nn.InstanceNorm3d(out_channel)
    if norm_name == "none":
        return nn.Sequential()
    raise NotImplementedError("The norm name is not recognized %s" % (norm_name))


def from_rgb(out_channels):
    #return sp_norm(nn.Conv3d(3, out_channels, (3, 3), padding=(1, 1), bias=True))
    return torch.nn.utils.spectral_norm(nn.Conv3d(3, out_channels, (3, 3, 3), padding=(1, 1, 1), bias=True))


def to_decision(out_channel, target_channel):
    #return sp_norm(nn.Conv3d(out_channel, target_channel, (1,1)))
    return torch.nn.utils.spectral_norm(nn.Conv3d(out_channel, target_channel, (1,1,1)))