import os
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random

class DFSDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--style_channel', type=int, default=6, help='# of style channels')
        parser.set_defaults(load_size=64, num_threads=4, display_winsize=64)
        if is_train:
            parser.set_defaults(display_freq=5120, update_html_freq=51200, print_freq=51200, save_latest_freq=5000000, n_epochs=10, n_epochs_decay=10, display_ncols=10)
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.style_channel = opt.style_channel
        self.img_size = opt.load_size
        self.phase = opt.phase
        self.table = {'english':'chinese', 'chinese':'english'}
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5), std = (0.5))])
        
        self.dataroot = os.path.join(opt.dataroot, opt.phase, opt.direction.split('2')[1])  # get the image directory
        self.paths = sorted(make_dataset(self.dataroot, opt.max_dataset_size))  # get image paths

    def __getitem__(self, index):
        # get image paths
        gt_path = self.paths[index]
        parts = gt_path.split(os.sep)
        IR_paths, characters = self.get_IR_paths(parts)
        CR_paths = self.get_CR_paths(parts, characters)
        CT_path = self.get_CT_path(parts)
        
        # load and transform images
        gt_image = self.load_image(gt_path)
        CT_image = self.load_image(CT_path)
        IR_images = torch.stack([self.load_image(IR_path) for IR_path in IR_paths], 0)
        CR_images = torch.stack([self.load_image(CR_path) for CR_path in CR_paths], 0)
        return {'gt_images':gt_image, 
                'CT_images':CT_image, 
                'IR_images':IR_images, 
                'CR_images':CR_images,
                'image_paths':gt_path}
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
    
    def load_image(self, path):
        image = Image.open(path)
        image = self.transform(image)
        return image
        
    def get_IR_paths(self, parts):
        IR_font_path = os.path.join(parts[0], parts[1], parts[2], parts[3], self.table[parts[4]], parts[5])
        characters = random.sample(os.listdir(IR_font_path), self.style_channel)
        IR_paths = [os.path.join(IR_font_path, character) for character in characters]
        return (IR_paths, characters)
    
    def get_CT_path(self, parts):
        return os.path.join(parts[0], parts[1], parts[2], parts[3], 'source', parts[-1])
    
    def get_CR_paths(self, parts, characters):
        return [os.path.join(parts[0], parts[1], parts[2], parts[3], 'source', character) for character in characters]