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
            parser.set_defaults(batch_size=512, display_freq=51200, update_html_freq=51200, print_freq=51200, save_latest_freq=512000, n_epochs=10, n_epochs_decay=10, display_ncols=10)
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = os.path.join(opt.dataroot, opt.phase, 'chinese_light')  # get the image directory
        self.paths = sorted(make_dataset(self.dataroot, opt.max_dataset_size))  # get image paths
        self.style_channel = opt.style_channel
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5), std = (0.5))])
        self.img_size = opt.load_size
        
    def __getitem__(self, index):
        # get chinese path and corresbonding english paths
        chinese_path = self.paths[index]
        english_paths = self.get_english_paths(chinese_path)
        # load and transform images
        gt_image, content_image = self.load_image(chinese_path)
        style_image = [self.load_image(english_path, mode=None) for english_path in english_paths]
        style_image = torch.cat(style_image, 0)
        return {'gt_images':gt_image, 'content_images':content_image, 'style_images':style_image,
                'style_image_paths':english_paths, 'image_paths':chinese_path}
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
    
    def load_image(self, path, mode='crop'):
        image = Image.open(path)
        image = self.transform(image)
        if mode:
            gt_image = image[:, :self.img_size, :]
            content_image = image[:, self.img_size:, :]
            return (gt_image, content_image)
        else:
            return image
        
    def get_english_paths(self, chinese_path):
        parts = chinese_path.split(os.sep)
        english_font_path = os.path.join(parts[0], parts[1], parts[2], parts[3], 'english', parts[5])
        english_paths = [os.path.join(english_font_path, letter) for letter in random.sample(os.listdir(english_font_path), self.style_channel)]
        return english_paths