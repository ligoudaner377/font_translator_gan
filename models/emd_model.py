import torch
from .base_model import BaseModel
from . import networks

class EMDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        if is_train:
            parser.set_defaults(batch_size=50, pool_size=0)
        parser.set_defaults(netG='EMD', dataset_mode='font')        
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.model_names = ['G']
        self.style_channel = opt.style_channel
        self.netG = networks.define_G(self.style_channel, output_nc=1, ngf=64, netG='EMD', init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        
        if self.isTrain:
            self.visual_names = ['gt_images', 'generated_images']+['style_images_{}'.format(i) for i in range(self.style_channel)]            
            self.loss_names = ['L1', 'bottom']
            self.loss_bottom = torch.tensor(1.0, dtype=torch.float).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        else:
            self.visual_names = ['gt_images', 'generated_images']
            
    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        if not self.isTrain:
            self.image_paths = data['image_paths']
            
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images = self.netG(self.style_images, self.content_images)
        
    def compute_weight(self):
        gt_images = self.gt_images/2.0+0.5
        batch_size = gt_images.shape[0]
        black_pixels = gt_images<0.5
        num_black_pixels = torch.sum(black_pixels, dim=[1, 2, 3])+1
        new_tensor = torch.where(black_pixels, gt_images, torch.tensor(0.).to(self.device))
        mean_black_pixels = torch.sum(new_tensor, dim=[1, 2, 3])/num_black_pixels
        weight = torch.nn.functional.softmax(mean_black_pixels, dim=0)*batch_size/num_black_pixels
        return weight
    
    def optimize_parameters(self,):
        self.forward()
        self.optimizer_G.zero_grad() 
        
        # self.loss_L1 = torch.mean(torch.abs(self.generated_images-self.gt_images))
        weight = self.compute_weight()
        self.loss_L1 = torch.mean(torch.sum(torch.abs(self.generated_images-self.gt_images), dim=[1, 2, 3])*weight)
        self.loss_L1.backward()
        self.optimizer_G.step()
        
    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):
                setattr(self, 'style_images_{}'.format(i), torch.unsqueeze(self.style_images[:, i, :, :], 1))
            self.netG.train() 