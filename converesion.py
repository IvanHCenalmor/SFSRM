import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

'''Base class for all neural network modules'''
from torch.nn.parallel import DataParallel, DistributedDataParallel
# import models.networks as networks
# import models.lr_scheduler as lr_scheduler
# from .base_model import BaseModel
# from models.modules.loss import GANLoss
# from models.modules.MS_SSIM_L1 import MSSSIML1_Loss, SSIM
# from models.modules.FFT_loss import FFTLoss

import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch

from collections import Counter
from collections import defaultdict
from torch.optim.lr_scheduler import _LRScheduler

##### MS-SSIM-L1 loss definition #####
#

def _fspecial_gauss_1d(size, sigma):
    """Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    """ Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

def _ssim(X, Y,
          data_range,
          win,
          size_average=True,
          K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs

def ms_ssim(X, Y,
            data_range=1,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            win=None,
            weights=None,
            K=(0.01, 0.03)):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        print(X.shape)
        print(Y.shape)
        raise ValueError('Input images should have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4) , \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y,
                                     win=win,
                                     data_range=data_range,
                                     size_average=False,
                                     K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    #CR
    # ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    ms_ssim_val = mcs_and_ssim ** weights.view(-1, 1, 1)
    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)

def gauss_weighted_l1(X, Y, win=None, win_size=33, win_sigma=8, size_average=False):
    '''
    (_, channel, height, width) = X.size()
    if win is None:
        win_x = torch.tensor([[2,-1,-1,-1,-1],[-1,2,-1,-1,-1],[-1,-1,2,-1,-1],[-1,-1,-1,2,-1],[-1,-1,-1,2,-1]])         
        win_x = win_x.repeat(X.shape[1], 1, 1, 1)
        win_y = torch.tensor([[-1,-1,-1,-1,2],[-1,-1,-1,2,-1],[-1,-1,2,-1,-1],[-1,2,-1,-1,-1],[2,-1,-1,-1,-1]])
        win_y = win_y.repeat(Y.shape[1], 1, 1, 1)
    win_x = win_x.to(X.device, dtype=X.dtype)
    win_y = win_y.to(X.device, dtype=X.dtype)
    x_Edage_x = F.conv2d(X, win_x, stride=1, padding=0, groups=1)
    x_Edage_y = F.conv2d(Y, win_x, stride=1, padding=0, groups=1)
    y_Edage_x = F.conv2d(X, win_y, stride=1, padding=0, groups=1)
    y_Edage_y = F.conv2d(Y, win_y, stride=1, padding=0, groups=1)
    l1 = (abs(x_Edage_x - x_Edage_y) + abs(y_Edage_x - y_Edage_y)) * 0.5
    if size_average:
        return l1.mean()
    else:
        return l1
    '''
    diff = abs(X - Y)
    (_, channel, height, width) = diff.size()
    if win is None:
        real_size = min(win_size, height, width)
        win = _fspecial_gauss_1d(real_size, win_sigma)
        win = win.repeat(diff.shape[1], 1, 1, 1)

    win = win.to(diff.device, dtype=diff.dtype)
    l1 = gaussian_filter(diff, win)
    if size_average:
        return l1.mean()
    else:
        return l1

class MSSSIML1_Loss(torch.nn.Module):
    def __init__(self,
                 data_range=1,
                 size_average=False,
                 win_size=3,
                 win_sigma=0.1,
                 channel=1,
                 weights=None,
                 K=(0.01, 0.03),
                 alpha=0.5):
        """ class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MSSSIML1_Loss, self).__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = _fspecial_gauss_1d(self.win_size, self.win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.alpha = alpha
        self.threshold = 0.05

    def forward(self, X, Y):
         #X = F.relu(X-self.threshold)
         #Y = F.relu(Y-self.threshold)
         ms_ssim_map = ms_ssim(X, Y,
                       data_range=self.data_range,
                       size_average=False,
                       win=self.win,
                       weights=self.weights,
                       K=self.K)
         l1_map = gauss_weighted_l1(X, Y,
                                    win=None,
                                    win_size=self.win_size,
                                    win_sigma=self.win_sigma,
                                    size_average=True)
         
         #CR loss_map = (1-ms_ssim_map) * alpha + l1_map * (1-alpha)
         '''CR: seems just a bias difference while no element difference related to img'''
         loss_map = (1 - ms_ssim_map) * self.alpha + l1_map * (1 - self.alpha)
         return loss_map.mean()

#
##### ############################### #####

##### FFT loss definition #####
#

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad,\
            "nn criterions don't compute the gradient w.r.t targets"

def FFT_trans(x, eps):
    vF = torch.fft.fft2(x, norm='ortho')
    vF = torch.fft.fftshift(vF)
    vF = torch.stack([vF.real, vF.imag], -1)
    # get real part
    #vR = vF[:,:,33:33+192-1,33:33+192-1, 0]
    # get the imaginary part
    vF[:,:,96:96+64-1,96:96+64-1, :] = 0
    #vI = vF - vF[:,:,33:33+192-1,33:33+192-1, 1]
    #vR = vF - vF[:,:,33:33+192-1,33:33+192-1, 0]
    #vI = vF[:,:,65:193,65:193, 1]
    vR = vF[:,:,:,:, 0]
    vI = vF[:,:,:,:, 1]
    out_amp = torch.add(torch.pow(vR, 2), torch.pow(vI, 2))
    out_amp = torch.sqrt(out_amp + eps)
    out_pha = torch.atan2(vR,(vI + eps))
    return out_amp, out_pha


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.eps = 1e-7


    def forward(self, SR, GT):
        _assert_no_grad(GT)
        real_fft_amp,  real_fft_pha = FFT_trans(GT,self.eps)
        fake_fft_amp,  fake_fft_pha = FFT_trans(SR,self.eps)
        amp_dis = real_fft_amp - fake_fft_amp
        pha_dis = real_fft_pha - fake_fft_pha
        fft_dis = (torch.pow(amp_dis,2) + torch.pow(pha_dis,2) + self.eps).sqrt()  #  +
        fftloss = fft_dis.mean()
        return fftloss

#
##### ############################### #####

##### GAN loss definition #####
#


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        #print('#######target_label.shape#########')
        #print(target_label.shape)
        loss = self.loss(input, target_label)
        return loss

#
##### ############################### #####

##### MultiStepLR_Restart definition #####
#

class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

#
##### ############################### #####

class SRGANModel():
    def __init__(self, 
                 gpu_ids=None,
                 is_train=True,
                 distributed_training=False,
                 gan_weight=1e-3,
                 pixel_weight=1,
                 pixel_criterion='ms_ssim_l1',
                 fft_weight=0.01,
                 fm_weight=0,
                 fm_criterion='l1',
                 feature_weight=0.01,
                 feature_criterion='l1',
                 gan_type='ragan',
                 D_update_ratio=1,
                 D_init_iters=0,
                 lr_G=1e-5,
                 weight_decay_G=0,
                 beta1_G=0.9,
                 beta2_G=0.99,
                 weight_decay_D=0,
                 lr_D=1e-5,
                 beta1_D=0.9,
                 beta2_D=0.99,
                 lr_steps=[50000, 100000, 200000, 300000],
                 lr_gamma=0.5,
                 pretrain_model_G_path=None,
                 pretrain_model_D_path=None,
                 save_path=None
                 ):
        
        self.device = torch.device('cuda' if gpu_ids is not None else 'cpu')
        
        print("##################")
        print(self.device)
        self.is_train = is_train
        self.schedulers = []
        self.optimizers = []

        self.pretrain_model_G_path = pretrain_model_G_path
        self.pretrain_model_D_path = pretrain_model_D_path
        self.save_path = save_path

        if distributed_training:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define networks and load pretrained models
        
        # Define the generator network
        opt_net = {'in_nc': 2, 'out_nc': 1, 'nf': 64, 'nb': 23}
        self.netG = RRDBNet_arch.RRDBNet(in_nc= opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                         nf=opt_net['nf'], nb=opt_net['nb']).to(self.device)
        
        if distributed_training:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            # Define the discriminator network
            self.netD = SRGAN_arch.UnetD().to(self.device)
            if distributed_training:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            self.netG.train()
            if gan_weight > 0:
                self.netD.train()
    

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if pixel_weight > 0:
                l_pix_type = pixel_criterion
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'ms_ssim_l1':
                    self.cri_pix = MSSSIML1_Loss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = pixel_weight
            else:
                self.cri_pix = None

            # G FFT loss
            if fft_weight > 0:
                self.cri_fft = FFTLoss().to(self.device)
                self.l_fft_w = fft_weight
            else:
                self.cri_fft = None

            # G Fm loss
            if fm_weight > 0:
                l_fm_type = fm_criterion
                if l_fm_type == 'l1':
                    self.cri_fm = nn.L1Loss().to(self.device)
                elif l_fm_type == 'l2':
                    self.cri_fm = nn.MSELoss().to(self.device)
                elif l_fm_type == 'ms_ssim_l1':
                    self.cri_fm = MSSSIML1_Loss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fm_type))
                self.l_fm_w = fm_weight
            else:
                self.cri_fm = None

            # G feature loss
            if feature_weight > 0:
                l_fea_type = feature_criterion
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'ssim_l1':
                    self.cri_fea = MSSSIML1_Loss().to(self.device)    
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = feature_weight
            else:
                self.cri_fea = None

            if self.cri_fea:  # load VGG perceptual loss
                netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=34, use_bn=False,
                                          use_input_norm=True, device=self.device)
                netF.eval()
                self.netF = netF.to(self.device)
                
                if distributed_training:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(gan_type, 1.0, 0.0).to(self.device)
            self.l_gan_w = gan_weight
            self.cri_l1 = nn.L1Loss().to(self.device)
            self.cri_l2 = nn.MSELoss().to(self.device)
            # D_update_ratio and D_init_iters
            self.D_update_ratio = D_update_ratio if D_update_ratio else 1
            self.D_init_iters = D_init_iters if D_init_iters else 0

            # optimizers
            # G
            wd_G = weight_decay_G if weight_decay_G else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=lr_G,
                                                weight_decay=wd_G,
                                                betas=(beta1_G, beta2_G))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = weight_decay_D if weight_decay_D else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D,
                                                weight_decay=wd_D,
                                                betas=(beta1_D, beta2_D))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            for optimizer in self.optimizers:
                self.schedulers.append(
                    MultiStepLR_Restart(optimizer, lr_steps,
                                        restarts=None, #train_opt['restarts'],
                                        weights=None, #train_opt['restart_weights'],
                                        gamma=lr_gamma,
                                        clear_state=None,#train_opt['clear_state']
                                        ))

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LW'].to(self.device)  # LW
        if need_GT:
            self.var_H = data['GT'].to(self.device)
            self.var_H_3 = self.var_H.repeat(1, 3, 1, 1)
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)


    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        self.fake_H_3 = self.fake_H.repeat(1, 3, 1, 1)
        
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fft:  # fft loss
                l_g_fft = self.l_fft_w * self.cri_fft(self.fake_H, self.var_H)
                l_g_total += l_g_fft

            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H_3).detach()
                fake_fea = self.netF(self.fake_H_3)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
           
            if self.l_gan_w > 0:
                e_S, d_S, e_Ss, d_Ss = self.netD(self.fake_H)
                _, _, e_Hs, d_Hs = self.netD(self.var_ref)
                l_g_gan = self.l_gan_w * 0.5 * (self.cri_gan(e_S, True) + self.cri_gan(d_S, True))
                l_g_total += l_g_gan
        
                if self.cri_fm:
                    l_g_fms = 0
                    for f in range(6):
                        l_g_fms += self.cri_fm(e_Ss[f], e_Hs[f])
                        l_g_fms += self.cri_fm(d_Ss[f], d_Hs[f])
                    l_g_fm = l_g_fms / 6 * self.l_fm_w
                    l_g_total += l_g_fm

            l_g_total.backward()
            self.optimizer_G.step()


        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total_D = 0
        e_S, d_S, _, _ = self.netD(self.fake_H.detach())
        e_H, d_H, _, _ = self.netD(self.var_ref)
        l_d_real_e = self.cri_gan(e_H, True)
        l_d_fake_e = self.cri_gan(e_S, False)

        l_d_real_d = self.cri_gan(d_H, True)
        l_d_fake_d = self.cri_gan(d_S, False)
        l_d_real_D = l_d_real_e + l_d_real_d
        l_d_total_D += l_d_real_D

        fake_H_CutMix = self.fake_H.detach().clone()

        #probability of doing cutmix
        p_mix = step /100000 #100000
        if p_mix > 0.5:
            p_mix = 0.5

        if torch.rand(1) <= p_mix:
            #n_mix += 1
            r_mix = torch.rand(1)  # real/fake ratio

            #def rand_bbox(self, size, lam):
            B = fake_H_CutMix.size()[0]
            C = fake_H_CutMix.size()[1]
            W = fake_H_CutMix.size()[2]
            H = fake_H_CutMix.size()[3]

            cut_rat = np.sqrt(1. - r_mix)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

        #         # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            fake_H_CutMix[:, :, bbx1:bbx2, bby1:bby2] = self.var_H[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (fake_H_CutMix.size()[-1] * fake_H_CutMix.size()[-2]))

            e_mix, d_mix, _, _ = self.netD(fake_H_CutMix)
            mask_true = torch.zeros(B,C,W,H)
            mask_true[:, :, bbx1:bbx2, bby1:bby2]=1
            mask_false= 1 - mask_true
            self.mask_true = mask_true.to(self.device)
            self.mask_false = mask_false.to(self.device)
            l_d_fake_e = self.cri_gan(e_mix, False)
            #l_d_fake_d = self.cri_gan(((1-2*self.mask)*d_mix), False)
            l_d_fake_d_r = self.cri_gan( d_mix * (self.mask_true),True)
            l_d_fake_d_f = self.cri_gan( d_mix * (self.mask_false), False)
            l_d_fake_d = l_d_fake_d_r * (1 - r_mix) + l_d_fake_d_f * (r_mix)

            d_S[:, :, bbx1:bbx2, bby1:bby2] = d_H[:, :, bbx1:bbx2, bby1:bby2]

            loss_d_cons = self.cri_l1(d_mix, d_S)

            l_d_total_D += loss_d_cons
        #
        l_d_fake_D = l_d_fake_e + l_d_fake_d
        l_d_total_D += l_d_fake_D
        l_d_total_D.backward()
        self.optimizer_D.step()
        
        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fm:
                self.log_dict['l_g_fm'] = l_g_fm.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_fft:
                self.log_dict['l_g_fft'] = l_g_fft.item()
            if self.l_gan_w >0:
                self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real_D.item()
        self.log_dict['l_d_fake'] = l_d_fake_D.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def load(self):
        load_path_G = self.pretrain_model_G_path
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG, strict_load=True)
        load_path_D = self.pretrain_model_D_path
        if self.is_train and load_path_D is not None:
            self.load_network(load_path_D, self.netD, strict_load=True)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.l_gan_w >0:
            self.save_network(self.netD, 'D', iter_step)

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.save_path, 'models', save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.save_path, 'training_state', save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

class LWGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self,
                phase,
                dataroot_GT="./training_dataset/train/HR",
                dataroot_LQ="./training_dataset/train/edgemap",
                dataroot_WF="./training_dataset/train/WF",
                scale=4,
                GT_size=256,
                use_flip=True,
                use_rot=True,
                ):
        
        super(LWGTDataset, self).__init__()
        """
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')
        """
        self.data_type = 'img'

        self.dataroot_GT = dataroot_GT
        self.dataroot_LQ = dataroot_LQ
        self.dataroot_WF = dataroot_WF
        self.scale = scale
        self.GT_size = GT_size
        self.phase = phase
        self.use_flip = use_flip
        self.use_rot = use_rot

        self.paths_LQ, self.paths_WF, self.paths_GT = None, None, None
        self.sizes_LQ, self.sizes_WF, self.sizes_GT = None, None, None
        self.LQ_env, self.WF_env, self.GT_env = None, None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, self.dataroot_GT)
        self.paths_WF, self.sizes_WF = util.get_image_paths(self.data_type, self.dataroot_WF)
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, self.dataroot_LQ)
        assert self.paths_GT, 'Error: GT path is empty.'
        assert self.paths_WF, 'Error: WF path is empty.'
        assert self.paths_LQ, 'Error: LQ path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]
        if self.paths_LQ and self.paths_WF:
            assert len(self.paths_LQ) == len(
                self.paths_WF
            ), 'WF and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.dataroot_GT, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.dataroot_LQ, readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.scale
        GT_size = self.GT_size

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        # modcrop in the validation / test phase
        if self.phase != 'train':
            img_GT = util.modcrop(img_GT, scale)

        # get WF image
        WF_path = self.paths_WF[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_WF[index].split('_')]
        else:
            resolution = None
        img_WF = util.read_img(self.WF_env, WF_path, resolution)

        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)


        if self.phase == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                img_WF = util.imresize_np(img_GT, 1 / scale, True)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale
            WF_size = LQ_size

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            img_WF = img_WF[rnd_h:rnd_h + WF_size, rnd_w:rnd_w + WF_size, :]

            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            # augmentation - flip, rotate
            img_LQ, img_GT, img_WF = util.augment([img_LQ, img_GT, img_WF], 
                                                  self.use_flip,
                                                  self.use_rot)


        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_WF = torch.from_numpy(np.ascontiguousarray(np.transpose(img_WF, (2, 0, 1)))).float()


        img_LW = torch.cat((img_LQ, img_WF), 0)
        return {'LQ': img_LQ, 'GT': img_GT, 'WF': img_WF, 'LW': img_LW,  'LQ_path': LQ_path, 'GT_path': GT_path, 'WF_path': WF_path}
    
    def __len__(self):
        return len(self.paths_GT)

import torch
import torch.utils.data

def create_dataloader(dataset, 
                      phase,
                      distributed_training,
                      n_workers=6,
                      batch_size=5,
                      gpu_ids=None,
                      sampler=None):
    
    if phase == 'train':
        if distributed_training:
            world_size = torch.distributed.get_world_size()
            num_workers = n_workers
            assert batch_size % world_size == 0
            batch_size = batch_size // world_size
            shuffle = False
        else:
            num_gpus = len(gpu_ids) if gpu_ids is not None else 1
            num_workers = n_workers * len(num_gpus)
            batch_size = batch_size
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)

import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

class DistIterSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dsize = len(self.dataset)
        indices = [v % dsize for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

import os
import math
import argparse
import random
import logging

from utils import util
import torch.distributed as dist
import torch.multiprocessing as mp

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():

    gpu_ids = '0'

    pretrain_model_G_path=None
    pretrain_model_D_path=None
    save_path=None
    val_images_path = os.path.join(save_path, 'val_images')

    distributed_training=False
    name = 'train'
    warmup_iter =  -1
    scale = 4
    batch_size = 5
    niter = 400000
    val_freq = 100
    print_freq: 100
    save_checkpoint_freq: 100
    use_tb_logger = True


    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        rank = -1
        print('Disabled distributed training.')
    else:
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(save_path)  # rename experiment folder if exists
            util.mkdirs(os.path.join(save_path, 'models'))
            util.mkdirs(os.path.join(save_path, 'training_state'))
            util.mkdirs(val_images_path)
            
        # config loggers. Before it, the log will not work
        util.setup_logger('base', save_path, 'train_' + name, level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', save_path, 'val_' + name, level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        # tensorboard logger
        if use_tb_logger and 'debug' not in name:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + name)
    else:
        util.setup_logger('base', save_path, 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    #### random seed
    seed = 666
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch

    train_set = LWGTDataset(dataset_opt)
    train_size = int(math.ceil(len(train_set) / batch_size))
    total_iters = int(niter)
    total_epochs = int(math.ceil(total_iters / train_size))
    if distributed_training:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
        total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
    else:
        train_sampler = None
    train_loader = create_dataloader(train_set, 
                                    'train',
                                     distributed_training,
                                     train_sampler)
    if rank <= 0:
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
            len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
            total_epochs, total_iters))

    val_set = LWGTDataset(dataset_opt)
    val_loader = create_dataloader(val_set, 
                                   'val',
                                   distributed_training, 
                                   None)
                
    assert train_loader is not None

    #### create model
    model = SRGANModel(    
        gpu_ids=gpu_ids,
        pretrain_model_G_path=pretrain_model_G_path,
        pretrain_model_D_path=pretrain_model_D_path,
        save_path=save_path
    )

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if distributed_training:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=warmup_iter)

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % print_freq == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if use_tb_logger and 'debug' not in name:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % val_freq == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_dis = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['WF_path'][0]))[0]
                    img_dir = os.path.join(val_images_path, img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = scale
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    'CR'
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    #avg_dis += util.cal_Hausdorff(sr_img,gt_img)

                avg_psnr = avg_psnr / idx
                #avg_dis = avg_dis / idx

                # log
                #logger.info('# Validation # PSNR: {:.4e} avg_dis: {:.4e}'.format(avg_psnr, avg_dis))
                logger.info('# Validation # PSNR: {:.4e} '.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                #logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, avg_dis: {:.4e}'.format(
                    #epoch, current_step, avg_psnr, avg_dis))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if use_tb_logger and 'debug' not in name:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    #tb_logger.add_scalar('avg_dis', avg_dis, current_step)

            #### save models and training states
            if current_step % save_checkpoint_freq == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()