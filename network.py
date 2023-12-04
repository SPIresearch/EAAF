from multiprocessing import reduction
from pickletools import optimize
from pyexpat import model
from tracemalloc import start
from unittest import result
import numpy as np
import torch,pdb,copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch.nn.functional as F
from loss import ce_loss,KL,entropy_loss, entropy_loss1,total_entropy_loss,CrossEntropy1,KL_KD,CrossEntropy2
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



# res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
# "resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}
res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        #self.linear=MyLinear(self.in_features)
        self.softplus=nn.Softplus()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
       
        return x
       

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
    def forward(self, x):
        x = self.fc(x)
 
        return x
    def forward(self, x):
        x = self.fc(x)
 
        return x

class evidence_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(evidence_classifier, self).__init__()
        self.type = type
        self.activation = nn.Softplus()
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        x=self.activation(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc
        self.softplus=nn.Softplus()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        x = self.softplus(x)
        return x, y




class MDDN(nn.Module):

    def __init__(self, mddn_f_list,mddn_C1_list,mddn_C2_list,mddn_E1_list,mddn_E2_list,classes, source, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param source: Number of source
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(MDDN, self).__init__()
        self.source = source
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.mddn_f=nn.ModuleList()
        self.mddn_C1=nn.ModuleList()
        self.mddn_C2=nn.ModuleList()
        self.mddn_E1=nn.ModuleList()
        self.mddn_E2=nn.ModuleList()
       
        for i in range(self.source):
            self.mddn_f.append(mddn_f_list[i])
            self.mddn_C1.append(mddn_C1_list[i])
            self.mddn_C2.append(mddn_C2_list[i])
            self.mddn_E1.append(mddn_E1_list[i])
            self.mddn_E2.append(mddn_E2_list[i])
       

   
    def update_batch_stats(self, flag):
        for v_num in range(self.source):
            for m in self.mddn_f[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
            for m in self.mddn_C1[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
          
            for m in self.mddn_C2[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
            for m in self.mddn_E2[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
          
           
    
    def sample_label(self,evidence,k=1):
        evidence=np.array(evidence.detach().cpu())
        sampled_labels=dict()
        alpha=evidence+1
        for z in range(k):
            sampled_labels[z]=[]
        for i in range(evidence.shape[0]):
            label=np.random.dirichlet(alpha[i],k)
            #label=np.argmax(label,1)
            for z in range(k):
                sampled_labels[z].append(label[z])
        for z in range(k):
            sampled_labels[z]=torch.from_numpy(np.array(sampled_labels[z])) 
        return sampled_labels

    def infer(self, input,v_num,mode,input_mode='x'):
        """
        :param input: Multi-source model
        :return: evidence of every model
        """
        
        if input_mode=='aug':
            self.update_batch_stats(False)
        
        if mode=='train':
            self.mddn_f[v_num].train()
            self.mddn_C1[v_num].train()
            self.mddn_E1[v_num].train()
            #self.mddn_C2[v_num].train()
        elif mode=='test':
            self.mddn_f[v_num].eval()
            self.mddn_C1[v_num].eval() 
            self.mddn_C2[v_num].eval() 
        f=self.mddn_f[v_num](input)
        feature=self.mddn_C1[v_num](f)
        evidence=self.mddn_E2[v_num](self.mddn_E1[v_num](f))

        out=self.mddn_C2[v_num](feature)
        
            #prob = nn.Softmax(1)(out)
        if input_mode=='aug':
            self.update_batch_stats(True)        
        return out,evidence,feature
    
    
    def forward_idx(self, input,mode,input_mode='x',v_num=0):
        self.out=dict()
        self.features=dict()   
        self.evi=dict() 
        self.out[v_num],self.evi[v_num],self.features[v_num]= self.infer(input,v_num,mode,input_mode)
           
      
        return self.out,self.features

    def forward(self, input,mode,idx=-1,input_mode='x'):
        self.out=dict()
        self.features=dict()
        self.evi=dict()
        self.alpha=dict()
        for v_num in range(self.source):   
            if v_num==idx:
                continue  
            self.out[v_num],self.evi[v_num],self.features[v_num] = self.infer(input,v_num,mode,input_mode)
            self.alpha[v_num] = self.evi[v_num] + 1
     
        return self.evi,self.out,self.features

 
