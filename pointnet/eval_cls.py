import numpy as np
import argparse
import torch
from models import cls_model
from utils import create_dir
from data_loader import get_data_loader
from collections import Counter

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    # directories 
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    return parser

if __name__ == '__main__':
    
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(args.device)
    
    create_dir(args.output_dir)
    
    # evaluating classification model 
    model = cls_model().to(args.device)
    
    # load best model 
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    # default: each cloud has 10000 points 
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # print(Counter(np.load(args.test_label)))
    # Counter({0.0: 617, 2.0: 234, 1.0: 102})
    
    ints = np.arange(685, 686)
    test_data = test_data[ints,:,:].to(args.device)
    test_label = test_label[ints].to(args.device).to(torch.long)

    # make predictions and compare with actual 
    with torch.no_grad():
        predictions = model(test_data)
        pred_label = torch.argmax(predictions, 1)    
    
    print(ints)
    print(pred_label)
    print(test_label)
    
    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))
