from evaluator.dataset import ClassifierDataset
from evaluator.classifier import Classifier
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import DataLoader

def train_classifier(mode='style', epochs=10):
    dataset = ClassifierDataset(mode)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)
    net = Classifier(mode, dataset.num_classes, isTrain=True, epochs=epochs)
    print('total images:{}'.format(len(dataset)))
    for epoch in range(epochs):
        start = time.time()
        for i, data in enumerate(dataloader):
            #####train####
            net.train(data)
            ##############
            if i%300==0:  # print loss for every 300 steps
                print('epoch: {:02}, step: {:05} loss:{:.4f}'.format(epoch, i, net.get_current_loss()))
        end = time.time()
        print('time for last epoch: {:.2f} min'.format((end-start)/60))
        net.save_networks(epoch+1)  # save the network every epoch
        net.update_learning_rate()  # update learning rate
    net.save_networks('latest')

    