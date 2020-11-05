import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class ClassifierDataset(Dataset):
    def __init__(self, mode):   
        path_chinese1 = 'datasets/font/train/chinese'
        path_english1 = 'datasets/font/train/english'
        path_chinese2 = 'datasets/font/test_unknown_style/chinese'
        path_english2 = 'datasets/font/test_unknown_style/english'
        path_chinese3 = 'datasets/font/test_unknown_content/chinese'
        roots = [path_chinese1, path_english1, path_chinese2, path_english2, path_chinese3]         
        if mode=='style': # build a style classifier
            part = -2     # choose which part as label
        elif mode=='content':
            part = -1            
        else:
            raise
        all_image_paths = self.get_all_paths(roots)
        classes = sorted(list(set(path.split(os.sep)[part] for path in all_image_paths)))
        self.text2label = dict(zip(classes, 
                              [i for i in range(len(classes))]))
        self.num_classes = len(classes)
        self.label2text = dict(zip([i for i in range(len(classes))],
                                     classes))        
        self.data = list(zip(all_image_paths, 
                        [self.text2label[path.split(os.sep)[part]] for path in all_image_paths]))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5), std = (0.5))])
        
    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = self.load_image(path)
        return (image, label)
    
    def __len__(self):   
        return len(self.data)
    
    def get_all_paths(self, roots):
        res = []
        if not isinstance(roots, list):
            roots = [roots]
        for root in roots:
            fonts = os.listdir(root)
            for font in fonts:
                font_path = os.path.join(root, font)
                characters = os.listdir(font_path)
                character_paths = [os.path.join(font_path, character) for character in characters]
                res.extend(character_paths)
        return res
    
    def load_image(self, path):
        image = Image.open(path).convert('L')
        image = self.transform(image)
        return image
    
   