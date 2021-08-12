import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
from torch.autograd import Variable
import functools
import numpy as np

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG== 'EMD':
        net = EMD_Generator(input_nc)
    elif netG=='DFS':
        net = DFS_Generator()
    elif netG=='FTGAN_CAT':
        net = FTGAN_Generator_CAT(style_channels=input_nc-1, ngf=ngf, use_dropout=use_dropout, n_blocks=6)
    elif netG=='FTGAN_AVG':
        net = FTGAN_Generator_AVG(ngf=ngf, use_dropout=use_dropout, n_blocks=6)
    elif netG=='FTGAN_HAN':
        net = FTGAN_Generator_HAN(ngf=ngf, use_dropout=use_dropout, n_blocks=6)
    elif netG=='FTGAN_MLAN':
        net = FTGAN_Generator_MLAN(ngf=ngf, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], use_spectral_norm=False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier Receptive Field = 70
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_64':  # default PatchGAN classifier Receptive Field = 34
        if use_spectral_norm:
            net = NLayerDiscriminatorS(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
        else:
            net = NLayerDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, train_gen=False):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if train_gen:
                loss = -prediction.mean()
            else:
                if target_is_real:
                    loss = torch.nn.ReLU()(1.0 - prediction).mean()
                else:
                    loss =  torch.nn.ReLU()(1.0 + prediction).mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
            
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        """Standard forward"""
        return self.model(torch.cat(inp, dim=1))

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
        
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)
                        ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                    ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
class NLayerDiscriminatorS(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminatorS, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                        spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)
                        ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                    ]
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)    
####### EMD Modules #####
class EMD_Encoder_Block(nn.Module):  
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding):
        super(EMD_Encoder_Block, self).__init__()
        self.encoder_block = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
                                           nn.BatchNorm2d(output_nc), 
                                           nn.LeakyReLU(negative_slope=0.2, inplace=True))
    def forward(self, x):
        return self.encoder_block(x)
    
class EMD_Decoder_Block(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, add, inner_layer=True):
        super(EMD_Decoder_Block, self).__init__()
        if inner_layer:
            self.decoder_block = nn.Sequential(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=add, bias=False),
                                               nn.BatchNorm2d(output_nc), 
                                               nn.ReLU(inplace=True))
        else:
            self.decoder_block = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=add, bias=False)
    def forward(self, x):
        return self.decoder_block(x)
    
class EMD_Content_Encoder(nn.Module):
    def __init__(self, content_channels=1):
        super(EMD_Content_Encoder, self).__init__()
        kernel_sizes = [5, 3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 2, 2]
        output_ncs = [64, 128, 256, 512, 512, 512, 512]
        for i in range(7):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            output_nc = output_ncs[i]
            input_nc = output_ncs[i-1] if i>0 else content_channels
            padding = kernel_size//2
            setattr(self, 'encoder_{}'.format(i), 
                    EMD_Encoder_Block(input_nc, output_nc, kernel_size, stride, padding))
    def forward(self, x):
        outps = [x]
        for i in range(7):
            outp = getattr(self, 'encoder_{}'.format(i))(outps[-1])
            outps.append(outp)
        return outps
    
class EMD_Style_Encoder(nn.Module):
    def __init__(self, style_channels):
        super(EMD_Style_Encoder, self).__init__()
        kernel_sizes = [5, 3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 2, 2]
        output_ncs = [64, 128, 256, 512, 512, 512, 512]
        for i in range(7):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            output_nc = output_ncs[i]
            input_nc = output_ncs[i-1] if i>0 else style_channels
            padding = kernel_size//2
            setattr(self, 'encoder_{}'.format(i), 
                    EMD_Encoder_Block(input_nc, output_nc, kernel_size, stride, padding))
    def forward(self, x):
        for i in range(7):
            x = getattr(self, 'encoder_{}'.format(i))(x)
        return x
    
class EMD_Decoder(nn.Module):
    def __init__(self,):
        super(EMD_Decoder, self).__init__()
        kernel_sizes = [3, 3, 3, 3, 3, 3, 5]
        strides = [2, 2, 2, 2, 2, 2, 1]
        output_ncs = [512, 512, 512, 256, 128, 64, 1]
        for i in range(7):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            output_nc = output_ncs[i]
            input_nc = output_ncs[i-1] if i>0 else 512
            padding = kernel_size//2
            add = stride//2
            setattr(self, 'decoder_{}'.format(i), 
                    EMD_Decoder_Block(input_nc*2, output_nc, kernel_size, stride, padding, add, inner_layer=(i<6)))
        self.out = nn.Tanh()
    def forward(self, x, layers):
        for i in range(7):
            x = torch.cat([x, layers[-i-1]], 1)
            x = getattr(self, 'decoder_{}'.format(i))(x)
        x = self.out(x)
        return x
    
class EMD_Mixer(nn.Module):
    def __init__(self, ):
        super(EMD_Mixer, self).__init__()
        self.mixer = nn.Bilinear(512, 512, 512)
        
    def forward(self, content_feature, style_feature):
        content_feature = torch.squeeze(torch.squeeze(content_feature, -1), -1)
        style_feature = torch.squeeze(torch.squeeze(style_feature, -1), -1)
        mixed = self.mixer(content_feature, style_feature)
        return torch.unsqueeze(torch.unsqueeze(mixed, -1), -1)
    
class EMD_Generator(nn.Module):
    def __init__(self, style_channels, content_channels=1):
        super(EMD_Generator, self).__init__()
        self.style_encoder = EMD_Style_Encoder(style_channels)
        self.content_encoder = EMD_Content_Encoder(content_channels)
        self.decoder = EMD_Decoder()
        self.mixer = EMD_Mixer()
        
    def forward(self, style_images, content_images):
        style_feature = self.style_encoder(style_images)
        content_features = self.content_encoder(content_images)
        mixed = self.mixer(content_features[-1], style_feature)
        generated = self.decoder(mixed, content_features)       
        return generated
    
#### DFS Modules ####
class DFS_Encoder(nn.Module):    
    def __init__(self, norm_layer=nn.BatchNorm2d, padding_type='reflect', mode='content'):
        super(DFS_Encoder, self).__init__()
        self.mode = mode
        self.in_pad = nn.ReflectionPad2d(3)
        self.in_conv = nn.Conv2d(1, 64, kernel_size=7, padding=0, bias=False)
        self.in_norm = norm_layer(64)
        self.in_nonlinear = nn.ReLU(True)
        
        self.down_conv_1 = nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_norm_1 = norm_layer(192)
        self.down_nonlinear_1 = nn.ReLU(True)
        
        self.down_conv_2 = nn.Conv2d(192, 576, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_norm_2 = norm_layer(576)
        self.down_nonlinear_2 = nn.ReLU(True)
    
        for i in range(6):       # add ResNet blocks
            setattr(self, 'resnet_block_{}'.format(i), ResnetBlock(576, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True, use_bias=False))
            
    def forward(self, inp):
        in_layer = self.in_norm(self.in_conv(self.in_pad(inp))) # 64
        g = self.in_nonlinear(in_layer)
        d1 = self.down_norm_1(self.down_conv_1(g)) # 32
        g = self.down_nonlinear_1(d1)
        d2 = self.down_norm_2(self.down_conv_2(g)) # 16
        g = self.down_nonlinear_2(d2)        
        for i in range(6):
            g = getattr(self, 'resnet_block_{}'.format(i))(g)
        if self.mode=='content':
            return (g, in_layer, d1, d2)
        else:
            return g
    
class DFS_Decoder(nn.Module):   
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(DFS_Decoder, self).__init__()
        self.up_conv_2 = nn.ConvTranspose2d(1728, 192, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.up_norm_2 = norm_layer(192)
        self.up_nonlinear_2 = nn.ReLU(True)
        
        self.up_conv_1 = nn.ConvTranspose2d(384, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.up_norm_1 = norm_layer(64)
        self.up_nonlinear_1 = nn.ReLU(True)
        
        self.out_pad = nn.ReflectionPad2d(3)
        self.out_conv = nn.Conv2d(128, 1, kernel_size=7, padding=0)
        self.out_nonlinear = nn.Tanh()
        
    def forward(self, inp):
        g, in_layer, d1, d2 = inp
        g = torch.cat([g, d2], 1)    
        g = self.up_nonlinear_2(self.up_norm_2(self.up_conv_2(g))) # 32     
        g= torch.cat([g, d1], 1)
        g = self.up_nonlinear_1(self.up_norm_1(self.up_conv_1(g))) # 64
        g = torch.cat([g, in_layer], 1)
        g = self.out_nonlinear(self.out_conv(self.out_pad(g)))
        return g
    
class DFS_Mixer(nn.Module):
    def __init__(self):
        super(DFS_Mixer, self).__init__()
        
    def forward(self, inp):
        IR_features, CR_features, CT_feature = inp
        
        # compute similarity matrix
        sim_mat = self.get_similarity(CT_feature, CR_features)
        sim_mat = F.softmax(sim_mat*2, dim=1) #the constant a is set to 2 to adjust their relative distance
        
        # compute style_feature by weighted sum
        style_feature = torch.sum(IR_features*sim_mat, dim=1)
        return style_feature
    
    def get_similarity(self, CT_feature, CR_features):
        CT_feature = torch.unsqueeze(CT_feature, dim=1)
        return torch.sum(CT_feature*CR_features, dim=[3, 4], keepdim=True)/torch.sqrt(torch.sum(CR_features**2, dim=[3, 4], keepdim=True))
    
class DFS_Generator(nn.Module):
    def __init__(self):
        super(DFS_Generator, self).__init__()
        self.content_encoder = DFS_Encoder(mode='content')
        self.style_encoder = DFS_Encoder(mode='style')
        self.mixer = DFS_Mixer()
        self.decoder = DFS_Decoder()
        
    def forward(self, inp):
        IR_images, CR_images, CT_image = inp
        K = IR_images.shape[1]
        
        # the error only passes back through the style feature side in the backprop
        self.set_requires_grad(self.style_encoder, False)
        CT_feature = self.style_encoder(CT_image)
        
        IR_features = []
        CR_features = []
        for i in range(K):
            CR_image = CR_images[:, i, :, :, :]
            CR_feature = self.style_encoder(CR_image)
            CR_features.append(CR_feature)   
        self.set_requires_grad(self.style_encoder, True)
        for i in range(K):
            IR_image = IR_images[:, i, :, :, :]
            IR_feature = self.style_encoder(IR_image)
            IR_features.append(IR_feature)
        IR_features = torch.stack(IR_features, dim=1)
        CR_features = torch.stack(CR_features, dim=1)
        
        content_feature, in_layer, d1, d2 = self.content_encoder(CT_image)
        style_feature = self.mixer((IR_features, CR_features, CT_feature))
        feature = torch.cat([content_feature, style_feature], dim=1)
        outp = self.decoder((feature, in_layer, d1, d2))
        return outp
    
    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
            
######### FTGAN ##########
class FTGAN_Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(FTGAN_Encoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]
        for i in range(2):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(ngf * mult * 2),
                      nn.ReLU(True)
                      ]
        self.model = nn.Sequential(*model)
        
    def forward(self, inp):
        return self.model(inp)
    
class FTGAN_Decoder(nn.Module):
    def __init__(self, use_dropout=False, n_blocks=6, ngf=64):
        super(FTGAN_Decoder, self).__init__()
        model = []
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf*8, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, use_bias=False)]
        for i in range(2):  # add upsampling layers
            mult = 2 ** (3 - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      nn.BatchNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf*2, 1, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        
    def forward(self, inp):
        return self.model(inp)
    
class FTGAN_Generator_AVG(nn.Module):
    def __init__(self, ngf=64, use_dropout=False, n_blocks=6):
        super(FTGAN_Generator_AVG, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf)
        
    def forward(self, inp):
        content_image, style_images = inp
        K = style_images.shape[1]
        content_feature = self.content_encoder(content_image)
        style_features = []
        for i in range(K):
            style_feature = self.style_encoder(torch.unsqueeze(style_images[:, i, :, :], dim=1))
            style_features.append(style_feature)
        style_features = torch.mean(torch.stack(style_features, dim=1), dim=1)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp
    
class FTGAN_Generator_CAT(nn.Module):
    def __init__(self, style_channels, ngf=64, use_dropout=False, n_blocks=9):
        super(FTGAN_Generator_CAT, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=style_channels, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf)
        
    def forward(self, inp):
        content_image, style_images = inp
        content_feature = self.content_encoder(content_image)
        style_features = self.style_encoder(style_images)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp   
    
class FTGAN_Local_Atten(nn.Module):
    def __init__(self, ngf=64):
        super(FTGAN_Local_Atten, self).__init__()
        self.self_atten = Self_Attn(ngf*4)
        self.attention = nn.Linear(ngf*4, 100)
        self.context_vec = nn.Linear(100, 1, bias=False)
        self.softmax  = nn.Softmax(dim=1) 
        
    def forward(self, style_features):
        B, C, H, W= style_features.shape
        h = self.self_atten(style_features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = torch.tanh(self.attention(h))                                   # (B*H*W, 100)
        h = self.context_vec(h)                                             # (B*H*W, 1)
        attention_map = self.softmax(h.view(B, H*W)).view(B, 1, H, W)       # (B, 1, H, W) 
        style_features = torch.sum(style_features*attention_map, dim=[2, 3])
        return style_features
    
class FTGAN_Global_Atten(nn.Module):
    def __init__(self, ngf=64):
        super(FTGAN_Global_Atten, self).__init__()
        self.ngf = ngf
        
    def forward(self, style_features, B, K):
        style_features = style_features.view(B, K, self.ngf*4)
        style_features = torch.mean(style_features, dim=1).view(B, self.ngf*4, 1, 1) # TBD
        style_features = style_features+torch.randn([B, self.ngf*4, 16, 16], device='cuda')*0.02
        return style_features
    
class FTGAN_Generator_HAN(nn.Module):
    def __init__(self, ngf=64, use_dropout=False, n_blocks=6):
        super(FTGAN_Generator_HAN, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf)
        self.local_atten = FTGAN_Local_Atten(ngf=ngf)
        self.global_atten = FTGAN_Global_Atten(ngf=ngf)
        
    def forward(self, inp):
        content_image, style_images = inp
        B, K, _, _ = style_images.shape
        content_feature = self.content_encoder(content_image)
        style_features = self.style_encoder(style_images.view(-1, 1, 64, 64))
        style_features = self.local_atten(style_features)
        style_features = self.global_atten(style_features, B, K)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp    
    
class FTGAN_Layer_Atten(nn.Module):
    def __init__(self, ngf=64):
        super(FTGAN_Layer_Atten, self).__init__()
        self.ngf = ngf
        self.fc = nn.Linear(4096, 3)
        self.softmax  = nn.Softmax(dim=1) 
        
    def forward(self, style_features, style_features_1, style_features_2, style_features_3, B, K):
        
        style_features = torch.mean(style_features.view(B, K, self.ngf*4, 4, 4), dim=1)
        style_features = style_features.view(B, -1)
        weight = self.softmax(self.fc(style_features))
        
        style_features_1 = torch.mean(style_features_1.view(B, K, self.ngf*4), dim=1)
        style_features_2 = torch.mean(style_features_2.view(B, K, self.ngf*4), dim=1)
        style_features_3 = torch.mean(style_features_3.view(B, K, self.ngf*4), dim=1)
        
        style_features = (style_features_1*weight.narrow(1, 0, 1)+
                          style_features_2*weight.narrow(1, 1, 1)+
                          style_features_3*weight.narrow(1, 2, 1)).view(B, self.ngf*4, 1, 1)+torch.randn([B, self.ngf*4, 16, 16], device='cuda')*0.02
        return style_features
    
class FTGAN_Generator_MLAN(nn.Module):
    def __init__(self, ngf=64, use_dropout=False, n_blocks=6):
        super(FTGAN_Generator_MLAN, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf)
        self.local_atten_1 = FTGAN_Local_Atten(ngf=ngf)
        self.local_atten_2 = FTGAN_Local_Atten(ngf=ngf)
        self.local_atten_3 = FTGAN_Local_Atten(ngf=ngf)
        self.layer_atten = FTGAN_Layer_Atten(ngf=ngf)
        
        self.downsample_1 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
                          nn.BatchNorm2d(ngf * 4),
                          nn.ReLU(True)
                          )
        self.downsample_2 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
                          nn.BatchNorm2d(ngf * 4),
                          nn.ReLU(True)
                          )
    def forward(self, inp):
        content_image, style_images = inp
        B, K, _, _ = style_images.shape
        content_feature = self.content_encoder(content_image)
        style_features = self.style_encoder(style_images.view(-1, 1, 64, 64))
        style_features_1 = self.local_atten_1(style_features)
        
        style_features = self.downsample_1(style_features)
        style_features_2 = self.local_atten_2(style_features)
        
        style_features = self.downsample_2(style_features)
        style_features_3 = self.local_atten_3(style_features)
        
        style_features = self.layer_atten(style_features, style_features_1, style_features_2, style_features_3, B, K)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp    
    
######### SA-GAN #########
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) 
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
