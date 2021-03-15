import torch
from .base_model import BaseModel
from . import networks

class FontTranslatorGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(norm='batch', netG='FTGAN_MLAN', dataset_mode='font')
        
        if is_train:
            parser.set_defaults(batch_size=256, pool_size=0, gan_mode='hinge', netD='basic_64')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
            parser.add_argument('--dis_2', default=True, help='use two discriminators or not')
            parser.add_argument('--use_spectral_norm', default=True)
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
            
        if self.isTrain:
            self.dis_2 = opt.dis_2
            self.visual_names = ['gt_images', 'generated_images']+['style_images_{}'.format(i) for i in range(self.style_channel)]
            if self.dis_2:
                self.model_names = ['G', 'D_content', 'D_style']
                self.loss_names = ['G_GAN', 'G_L1', 'D_content', 'D_style']
            else:
                self.model_names = ['G', 'D']
                self.loss_names = ['G_GAN', 'G_L1', 'D']
        else:
            self.visual_names = ['gt_images', 'generated_images']
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.style_channel+1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.dis_2:
                self.netD_content = networks.define_D(2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
                self.netD_style = networks.define_D(self.style_channel+1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            else:
                self.netD = networks.define_D(self.style_channel+2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            
        if self.isTrain:
            # define loss functions
            self.lambda_L1 = opt.lambda_L1
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.dis_2:
                self.lambda_style = opt.lambda_style
                self.lambda_content = opt.lambda_content
                self.optimizer_D_content = torch.optim.Adam(self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_style = torch.optim.Adam(self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_content)
                self.optimizers.append(self.optimizer_D_style)
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            
    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        if not self.isTrain:
            self.image_paths = data['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images = self.netG((self.content_images, self.style_images))
        
    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real = torch.cat(real_images, 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
    
    def compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.dis_2:
            self.loss_D_content = self.compute_gan_loss_D([self.content_images, self.gt_images],  [self.content_images, self.generated_images], self.netD_content)
            self.loss_D_style = self.compute_gan_loss_D([self.style_images, self.gt_images], [self.style_images, self.generated_images], self.netD_style)
            self.loss_D = self.lambda_content*self.loss_D_content + self.lambda_style*self.loss_D_style         
        else:
            self.loss_D = self.compute_gan_loss_D([self.content_images, self.style_images, self.gt_images], [self.content_images, self.style_images, self.generated_images], self.netD)

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.dis_2:
            self.loss_G_content = self.compute_gan_loss_G([self.content_images, self.generated_images], self.netD_content)
            self.loss_G_style = self.compute_gan_loss_G([self.style_images, self.generated_images], self.netD_style)
            self.loss_G_GAN = self.lambda_content*self.loss_G_content + self.lambda_style*self.loss_G_style
        else:
            self.loss_G_GAN = self.compute_gan_loss_G([self.content_images, self.style_images, self.generated_images], self.netD)
            
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], True)
            self.optimizer_D_content.zero_grad()
            self.optimizer_D_style.zero_grad()
            self.backward_D()
            self.optimizer_D_content.step()
            self.optimizer_D_style.step()
        else:
            self.set_requires_grad(self.netD, True)      # enable backprop for D
            self.optimizer_D.zero_grad()             # set D's gradients to zero
            self.backward_D()                    # calculate gradients for D
            self.optimizer_D.step()                # update D's weights
        # update G
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], False)
        else:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        self.backward_G()                             # calculate graidents for G
        self.optimizer_G.step()                       # udpate G's weights

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):
                setattr(self, 'style_images_{}'.format(i), torch.unsqueeze(self.style_images[:, i, :, :], 1))
            self.netG.train()
        else:
            pass    