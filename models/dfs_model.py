import torch
from .base_model import BaseModel
from . import networks

class DFSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(netG='DFS', dataset_mode='dfs')
        
        if is_train:
            parser.set_defaults(batch_size=64, pool_size=0, gan_mode='lsgan', netD='basic_64')
            parser.add_argument('--lambda_L1', type=float, default=0.5, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=0.5, help='weight for L2 loss')
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
        if self.isTrain:              
            self.model_names = ['G', 'D']
            self.loss_names = ['G_GAN', 'G_L1', 'G_L2','D']
        else:
            self.visual_names = ['gt_images', 'generated_images']
            self.model_names = ['G']
        
        self.netG = networks.define_G(1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:             
            self.netD = networks.define_D(1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.CT_images = data['CT_images'].to(self.device)
        self.CR_images = data['CR_images'].to(self.device)
        self.IR_images = data['IR_images'].to(self.device)
        if self.isTrain: 
            self.visual_names = ['gt_images', 'generated_images']+['IR_images_{}'.format(i) for i in range(self.IR_images.shape[1])]
        else:
            self.image_paths = data['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images = self.netG((self.IR_images, self.CR_images, self.CT_images))
        
    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        pred_fake = netD(fake_images.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = netD(real_images)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
    
    def compute_gan_loss_G(self, fake_images, netD):
        pred_fake = netD(fake_images)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        return loss_G_GAN
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D = self.compute_gan_loss_D(self.gt_images, self.generated_images, self.netD)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_GAN = self.compute_gan_loss_G(self.generated_images, self.netD)
            
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1
        self.loss_G_L2 = self.criterionL2(self.generated_images, self.gt_images) * self.opt.lambda_L2
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()             # set D's gradients to zero
        self.backward_D()                        # calculate gradients for D
        self.optimizer_D.step()                  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        self.backward_G()                             # calculate graidents for G
        self.optimizer_G.step()                       # udpate G's weights

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.IR_images.shape[1]):
                setattr(self, 'IR_images_{}'.format(i), self.IR_images[:, i, :, :, :])
            self.netG.train()