import os
from options.test_options import TestOptions
from evaluator import train_classifier
from evaluator.evaluator import EvaluatorDataset
from evaluator.evaluator import Evaluator
from evaluator.dataset import ClassifierDataset
from torch.utils.data import DataLoader

if __name__=="__main__":
    opt = TestOptions().parse() 
    classifier_path = 'evaluator/checkpoints/latest_{}_resnet.pth'.format(opt.evaluate_mode)    
    if not os.path.exists(classifier_path):   
        train_classifier(mode=opt.evaluate_mode)
    training_data = ClassifierDataset(opt.evaluate_mode)
    dataset = EvaluatorDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    evaluator = Evaluator(opt, num_classes=training_data.num_classes, text2label=training_data.text2label)
    
    for data in dataloader:  
        try:
            evaluator.evaluate(data)
            evaluator.record_current_results()
        except:
            print('!!!error!!! pass current data')
    evaluator.compute_final_results()