import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import cls_model
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir

def train(train_dataloader, model, opt, epoch, args, writer):
    
    model.train()
    step = epoch*len(train_dataloader)
    # print(len(train_dataloader))
    epoch_loss = 0
    correct_obj = 0 
    num_obj = 0

    for i, batch in enumerate(train_dataloader):
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)
        
        # run model 
        predictions = model(point_clouds)
        pred_labels = torch.argmax(predictions, 1)

        # cross entropy loss 
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        # print(loss.item())
        
        # compute running loss 
        epoch_loss += loss
        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('train_loss', loss.item(), step+i)
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        num_obj += labels.size()[0]

    # accuracy 
    epoch_acc = correct_obj / num_obj
    return epoch_loss, epoch_acc

def test(test_dataloader, model, epoch, args, writer):
    
    model.eval()
    test_loss = 0
    correct_obj = 0
    num_obj = 0
    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # make predictions 
        with torch.no_grad():
            predictions = model(point_clouds)
            pred_labels = torch.argmax(predictions, 1)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(predictions, labels)
            test_loss += loss
            
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        num_obj += labels.size()[0]

    #compute test accuracy 
    accuracy = correct_obj / num_obj

    writer.add_scalar("test_acc", accuracy, epoch)
    return test_loss, accuracy


def main(args):

    # save results for plots 
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    
    # create directories for saving results 
    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    # Tensorboard Logger
    writer = SummaryWriter('./logs/{0}'.format('cls'+"_"+args.exp_name))

    # init model 
    model = cls_model()
    model.to(args.device)
    
    # Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print ("successfully loaded checkpoint from {}".format(model_path))

    # optimizer 
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # dataloader 
    train_dataloader = get_data_loader(args=args, train=True)
    test_dataloader = get_data_loader(args=args, train=False)

    print ("successfully loaded data")

    best_acc = -1

    print ("======== start training for {} task ========".format('cls'))
    print ("(check tensorboard for plots of experiment logs/{})".format('cls'+"_"+args.exp_name))
    

    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss, train_epoch_acc = train(train_dataloader, model, opt, epoch, args, writer)
        train_loss_list.append(train_epoch_loss.item())
        train_acc_list.append(train_epoch_acc)
        # Test
        current_loss, current_acc = test(test_dataloader, model, epoch, args, writer)
        test_loss_list.append(current_loss.item())
        test_acc_list.append(current_acc)
        
        
        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))
        
        # save model after 10 epochs 
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # save the model with the best accuracy 
        if (current_acc >= best_acc):
            best_acc = current_acc
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)
            
    # open file in write mode
    with open(r'./train_loss_list.txt', 'w') as fp:
        for item in train_loss_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
        
    # open file in write mode
    with open(r'./train_acc_list.txt', 'w') as fp:
        for item in train_acc_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
        
    # open file in write mode
    with open(r'./test_loss_list.txt', 'w') as fp:
        for item in test_loss_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
        
    # open file in write mode
    with open(r'./test_acc_list.txt', 'w') as fp:
        for item in test_acc_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    print ("======== training completes ========")


def create_parser():
    # parser 
    parser = argparse.ArgumentParser()

    # hyperparameters 
    parser.add_argument('--task', type=str, default="cls")
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=8) # adjutsed for low cuda memory 
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp_name', type=str, default="exp")
    # saving directories 
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=10) 
    parser.add_argument('--load_checkpoint', type=str, default='')
    return parser

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(args.device)
    args.checkpoint_dir = args.checkpoint_dir+"/"+'cls' 

    main(args)