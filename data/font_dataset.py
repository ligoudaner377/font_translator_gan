import os
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random

class FontDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--style_channel', type=int, default=6, help='# of style channels')
        parser.set_defaults(load_size=64, num_threads=4, display_winsize=64)
        if is_train:
            parser.set_defaults(display_freq=51200, update_html_freq=51200, print_freq=51200, save_latest_freq=5000000, n_epochs=10, n_epochs_decay=10, display_ncols=10)
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if opt.direction=="english2chinese":
            self.content_language = 'chinese'
            self.style_language = 'english'
        else:
            self.content_language = 'english'
            self.style_language = 'chinese'
        BaseDataset.__init__(self, opt)
        self.dataroot = os.path.join(opt.dataroot, opt.phase, self.content_language)  # get the image directory
        self.paths = sorted(make_dataset(self.dataroot, opt.max_dataset_size))  # get image paths
        self.style_channel = opt.style_channel
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5,), std = (0.5,))])
        self.img_size = opt.load_size
        
    def __getitem__(self, index):
        # get content path and corresbonding stlye paths
        gt_path = self.paths[index]
        parts = gt_path.split(os.sep)
        style_paths = self.get_style_paths(parts)
        content_path = self.get_content_path(parts)
        # load and transform images
        content_image = self.load_image(content_path)
        gt_image = self.load_image(gt_path)
        style_image = torch.cat([self.load_image(style_path) for style_path in style_paths], 0)
        return {'gt_images':gt_image, 'content_images':content_image, 'style_images':style_image,
                'style_image_paths':style_paths, 'image_paths':gt_path}
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
    
    def load_image(self, path):
        image = Image.open(path)
        image = self.transform(image)
        return image
        
    def get_style_paths(self, parts):
        english_font_path = os.path.join(parts[0], parts[1], parts[2], parts[3], self.style_language, parts[5])
        english_paths = [os.path.join(english_font_path, letter) for letter in random.sample(os.listdir(english_font_path), self.style_channel)]
        return english_paths
    
    def get_content_path(self, parts):
        return os.path.join(parts[0], parts[1], parts[2], parts[3], 'source', parts[-1])
