import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# original PointNet code in TF: https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls_basic.py
# The authors made slight archiecture changes here: https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
# helpful: https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698

# Shared MLP = 1D convolution 
##### T-Net #####
class transform_net(nn.Module):
    
    # JEFF: input_size argument due to differences in input transformation and feature transformation
    def __init__(self, input_size): 
        super(transform_net, self).__init__()
        
        self.input = input_size
        
        self.conv1 = torch.nn.Conv1d(self.input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # FC layers: No dropout
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )        
        
        self.fc3 = nn.Linear(256, self.input * self.input)

    def forward(self, x):
        # The first transformation network is a mini-PointNet that takes raw 
        # point cloud as input and regresses to a 3 x 3 matrix.
        batch_size = x.size()[0]

        # All layers, except the last one, include ReLU and batch normalization.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # point features maxpool aggregation 
        # x = nn.MaxPool1d(batchsize)(x)
        # x = x.view(-1, 1024)
        # print(x.shape)
        x = torch.max(x, 2, keepdim=True)[0] # prevent dim loss
        x = x.view(-1, 1024)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # The output matrix is
        # initialized as an identity matrix. All layers, except the last
        # one, include ReLU and batch normalization.

        # JEFF: complete identity matrix, not 100% sure this is right
        # id_mat = 1

        # id_mat = torch.from_numpy(np.eye(self.input).flatten().astype(np.float32))
        # id_mat = id_mat.view(1, self.input * self.input).repeat(batch_size, 1)
        # x = x + id_mat.cuda()

        iden = torch.eye(self.input, requires_grad=True).repeat(batch_size, 1, 1)
        x = x.view(-1, self.input, self.input) + iden.cuda()

        # print(x.shape)
        return x # returns 3x3 or 64 x 64


##### Classification with Transformation #####
class cls_model_trans(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model_trans, self).__init__()
        
        # transformations
        self.in_trans = transform_net(input_size = 3)
        self.feat_trans = transform_net(input_size = 64)
        
        # normal network
        self.conv1 = nn.Conv1d(3, 64, 1) 
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        
        
        # FC layers
        # in pointnet_cls.py, they change this to have dropout... 
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )

        # original paper uses dropout in 2nd FC layer in basic implementation
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
            )        
        
        self.fc3 = nn.Linear(256, num_classes)
        
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Original PointNet Paper 
        # The first transformation network is a mini-PointNet that takes raw 
        # point cloud as input and regresses to a 3 x 3 matrix.
        
        x = points.transpose(2, 1) # B, 3, N
        
        # input transformation
        trans = self.in_trans(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans).transpose(2, 1)
        
        # All layers, except the last one, include ReLU and batch normalization.
        x = F.relu(self.bn1(self.conv1(x)))
        # input transformation
        trans = self.feat_trans(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans).transpose(2, 1)

        
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # point features maxpool aggregation 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


##### Vanilla Classification #####
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        # normal network
        self.conv1 = nn.Conv1d(3, 64, 1) 
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        
        
        # FC layers
        # in pointnet_cls.py, they change this to have dropout... 
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )

        # original paper uses dropout in 2nd FC layer in basic implementation
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
            )        
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Original PointNet Paper 
        # The first transformation network is a mini-PointNet that takes raw 
        # point cloud as input and regresses to a 3 x 3 matrix.

        x = points.transpose(2, 1) # B, 3, N
        
        # All layers, except the last one, include ReLU and batch normalization.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # point features maxpool aggregation 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x