import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from .ssim import SSIM, MSSSIM
from .fid import FID
from .classifier import Classifier
import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, opt, num_classes=None, text2label=None):
        self.text2label = text2label
        self.evaluate_mode = opt.evaluate_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out_root = os.path.join(opt.results_dir, opt.name, opt.phase+'_{}'.format(opt.epoch), 'metrics', opt.evaluate_mode)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionSSIM = SSIM().to(self.device)
        self.criterionMSSSIM = MSSSIM(weights=[0.45, 0.3, 0.25]).to(self.device)     
        self.criterionFID = FID(opt.evaluate_mode, num_classes, gpu_ids=opt.gpu_ids)
                
    def set_input(self, data):
        self.gt_images = data[0].permute([1, 0, 2, 3]).to(self.device)
        self.generated_images = data[1].permute([1, 0, 2, 3]).to(self.device)
        self.labels = data[2][0]
        
    def compute_l1(self):
        self.l1 = self.criterionL1(self.gt_images/2+0.5, self.generated_images/2+0.5).item()
    
    def compute_ssim(self):
        self.ssim = self.criterionSSIM(self.gt_images/2+0.5, self.generated_images/2+0.5).item()
    
    def compute_msssim(self):
        self.msssim = self.criterionMSSSIM(self.gt_images/2+0.5, self.generated_images/2+0.5).item()
        if np.isnan(self.msssim):
            self.msssim = 0.
            
    def compute_acc(self):                
        if self.evaluate_mode=='content':
            labels = self.text2label[self.labels+'.png']
        else:
            labels = self.text2label[self.labels]
        self.acc = self.criterionFID.get_acc(labels).item()
    
    def compute_fid(self):
        self.fid = self.criterionFID.forward(self.gt_images, self.generated_images)
                
    def evaluate(self, data):        
        self.set_input(data)
        
        self.compute_fid()        
        self.compute_acc()
        self.compute_l1()
        self.compute_ssim()
        self.compute_msssim()
        
            
    def get_current_results(self):
        return {'batch_size':self.gt_images.shape[0], 
                'l1':self.l1, 
                'ssim':self.ssim, 
                'msssim':self.msssim, 
                'fid': self.fid, 
                'num_correct':self.acc}
    
    def record_current_results(self):
        print('----------- current results -------------')
        print()
        print('label       :', self.labels)
        print('batch size  :', self.gt_images.shape[0])
        print('num_correct :', self.acc)
        print('l1          :', self.l1)
        print('ssim        :', self.ssim)
        print('msssim      :', self.msssim)
        print('fid         :', self.fid)
        print()
        res = [str(self.gt_images.shape[0])+'\n',
               str(self.acc)+'\n',
               str(self.l1)+'\n', 
               str(self.ssim)+'\n', 
               str(self.msssim)+'\n',
               str(self.fid)]
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)
        with open(os.path.join(self.out_root, self.labels)+'.txt', 'w') as f:
            f.writelines(res)
            
    def compute_final_results(self, ):
        num_images, num_correct, l1, ssim, msssim, fid = 0, 0, 0, 0, 0, 0
        files = os.listdir(self.out_root)
        print('loading metrics...')
        for file in files:
            if '.txt' not in file:
                continue
            if file!='final_results.txt':
                file_path = os.path.join(self.out_root, file)
                with open(file_path, 'r') as f:
                    l = f.read().split('\n')
                    num_images += int(l[0])
                    num_correct += int(l[1])
                    l1  += float(l[2])*int(l[0])
                    ssim += float(l[3])*int(l[0])
                    msssim += float(l[4])*int(l[0])
                    fid += float(l[5])*int(l[0])
        acc = num_correct/num_images
        l1 = l1/num_images
        ssim = ssim/num_images
        msssim = msssim/num_images
        fid = fid/num_images
        res = ['acc:'+str(acc)+'\n',
               'l1:'+str(l1)+'\n', 
               'ssim:'+str(ssim)+'\n', 
               'msssim:'+str(msssim)+'\n',
               'fid:'+str(fid)]
        with open(os.path.join(self.out_root, 'final_results.txt'), 'w') as f:
            f.writelines(res)
            print('results saved at {}'.format(os.path.join(self.out_root, 'final_results.txt')))
            
    def show_examples(self):
        idx = np.random.randint(0, self.gt_images.shape[0])
        plt.figure(figsize=[5, 10])
        plt.subplot(1, 2, 1)
        plt.imshow(self.gt_images.cpu()[idx, 0, :, :], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(self.generated_images.cpu()[idx, 0, :, :], cmap='gray')
        plt.axis('off')
        plt.show()
        
class EvaluatorDataset(Dataset):
    def __init__(self, opt):
        data_root = os.path.join(opt.results_dir, opt.name, opt.phase+'_{}'.format(opt.epoch), 'images')
        if opt.evaluate_mode=='content':
            part = 1
        elif opt.evaluate_mode=='style':
            part = 0
        else:
            raise
        all_image_paths = os.listdir(data_root)
        self.all_classes = list(set(path.split('|')[part] for path in all_image_paths))   
        self.table = {key:[[],[]] for key in self.all_classes}
        for path in all_image_paths:
            key = path.split('|')[part]
            category = path.split('|')[2]
            path = os.path.join(data_root, path)
            
            if category=='gt_images.png':
                self.table[key][0].append(path) 
            elif category=='generated_images.png':
                self.table[key][1].append(path)
            else:
                raise
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5), std = (0.5))])
    def __getitem__(self, idx):
        key = self.all_classes[idx]
        gt_images, generated_images = sorted(self.table[key][0]), sorted(self.table[key][1])
        gt_images = torch.cat([self.load_image(path) for path in gt_images], 0)
        generated_images = torch.cat([self.load_image(path) for path in generated_images], 0)        
        return (gt_images, generated_images, key)
    
    def __len__(self):
        return len(self.all_classes)
    
    def load_image(self, path):
        image = Image.open(path).convert('L')
        image = self.transform(image)
        return image