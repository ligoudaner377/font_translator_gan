from . import resnet
import torch.nn as nn
import torch.optim as optim
import torch
import os

class Classifier():
    def __init__(self, mode, num_classes, isTrain=False, save_dir='evaluator/checkpoints', gpu_ids=[0, 1], epochs=10):
        self.gpu_ids = gpu_ids
        self.mode = mode
        self.save_dir = save_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not num_classes:
            if mode=='style':
                num_classes = 847
            elif mode=='content':
                num_classes = 1074
            else:
                raise    
        self.resnet = resnet.resnet50(num_classes=num_classes)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.resnet.to(gpu_ids[0])
            self.resnet = torch.nn.DataParallel(self.resnet, gpu_ids)  # multi-GPUs
        if isTrain:            
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.resnet.parameters(), lr=0.001, betas=(0.5, 0.999))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                 lr_lambda=lambda epoch:1.0 - max(0, epoch+1-epochs//2)/float(epochs//2+1))
        else:
            self.load_networks('latest')            
            
    def set_input(self, data):
        self.images, self.labels = data[0].to(self.device), data[1].to(self.device)
        
    def forward(self):
        self.predicts, self.activations = self.resnet(self.images)
    
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss = self.criterion(self.predicts, self.labels)
        self.loss.backward()
        self.optimizer.step()
        
    def train(self, data):
        self.set_input(data)
        self.forward()
        self.optimize_parameters()
        
    def test(self, data): 
        self.set_input(data)
        self.resnet.eval()
        with torch.no_grad():
            self.forward()     
        
    def get_current_loss(self):
        return float(self.loss)
    
    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        
    def save_networks(self, epoch):
        save_filename = '%s_%s_resnet.pth' % (epoch, self.mode)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.resnet.module.cpu().state_dict(), save_path)
            self.resnet.cuda(self.gpu_ids[0])
        else:
            torch.save(self.resnet.cpu().state_dict(), save_path)
            
    def load_networks(self, epoch):
        load_filename = '%s_%s_resnet.pth' % (epoch, self.mode)
        load_path = os.path.join(self.save_dir, load_filename)
        net = self.resnet
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)
        print('%s classifier loaded!' % self.mode)
        
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
            