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

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

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
       
class feat_EC(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="ori"):
        super(feat_EC, self).__init__()
        self.ec = nn.Sequential(
                nn.Linear(bottleneck_dim, 256),
                nn.ReLU(),
                # 在第一个全连接层之后添加一个dropout层
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(),
                # 在第二个全连接层之后添加一个dropout层
                nn.Dropout(),
                nn.Linear(256, class_num))
        self.ec.apply(init_weights)
    def forward(self, x):
        x = self.ec(x)
        
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


class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(torch.tensor(1.)*init_weights)   
    
    def forward(self,x):
        x = self.w*torch.ones((x.shape[0]),1).cuda()
        x = torch.sigmoid(x)
        return x


class source_quantizer(nn.Module):
    def __init__(self, source_num, type="linear"):
        super(source_quantizer, self).__init__()
        self.type = type
        # self.quantizer = nn.Linear(source_num, 1)

        if type == 'wn':
            self.quantizer = weightNorm(nn.Linear(source_num, 1), name="weight")
            self.quantizer.apply(init_weights)
        else:
            self.quantizer = nn.Linear(source_num, 1)
            self.quantizer.apply(init_weights)

    def forward(self, x):
        x = self.quantizer(x)
        # x = torch.sigmoid(x)
        x = torch.softmax(x, dim=0)
        return x

class MyLinear(nn.Module):
    def __init__(self, in_units):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,in_units))
        #self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.mul(X, self.weight.data)
        return linear


class ME(nn.Module):

    def __init__(self, netF_list,netB_list,netC_list,netE_list,netEC_list,classes, source, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param source: Number of source
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(ME, self).__init__()
        self.source = source
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.netF=nn.ModuleList()
        self.netB=nn.ModuleList()
        self.netC=nn.ModuleList()
        self.netE=nn.ModuleList()
        self.netEC=nn.ModuleList()
       
        for i in range(self.source):
            self.netF.append(netF_list[i])
            self.netB.append(netB_list[i])
            self.netC.append(netC_list[i])
            self.netE.append(netE_list[i])
            self.netEC.append(netEC_list[i])
       

    def DS_Combin(self, alpha,idx):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2,weight):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            weight=torch.cat([weight,S_a.detach().clone()],1)
            return alpha_a,weight
        if idx==-1:

            for v in range(len(alpha)-1):
                if v==0:
                    S = torch.sum(alpha[0], dim=1, keepdim=True)
                    u= self.classes/S
                    weight=S
                    alpha_a,weight = DS_Combin_two(alpha[0], alpha[1],weight)
                else:
                    alpha_a,weight = DS_Combin_two(alpha_a, alpha[v+1],weight)
        elif idx==1:
            for v in range(len(alpha)-1):
                if v==idx:
                    continue
                if v==0:
                    S = torch.sum(alpha[0], dim=1, keepdim=True)
                    u= self.classes/S
                    weight=S
                    alpha_a,weight = DS_Combin_two(alpha[0], alpha[2],weight)
                else:
                    alpha_a,weight = DS_Combin_two(alpha_a, alpha[v+1],weight)
        elif idx==0:
            for v in range(1,len(alpha)-1):
                if v==1:
                    S = torch.sum(alpha[1], dim=1, keepdim=True)
                    u= self.classes/S
                    weight=S
                    alpha_a,weight = DS_Combin_two(alpha[1], alpha[2],weight)
                else:
                    alpha_a,weight = DS_Combin_two(alpha_a, alpha[v+1],weight)
        else:
            for v in range(len(alpha)-1):
                if v==idx:
                    continue
                if v==0:
                    S = torch.sum(alpha[0], dim=1, keepdim=True)
                    u= self.classes/S
                    weight=S
                    alpha_a,weight = DS_Combin_two(alpha[0], alpha[1],weight)
                else:
                    alpha_a,weight = DS_Combin_two(alpha_a, alpha[v+1],weight)
     
        return alpha_a
    def update_batch_stats(self, flag):
        for v_num in range(self.source):
            for m in self.netF[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
            for m in self.netB[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
          
            for m in self.netC[v_num].modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.update_batch_stats = flag
            for m in self.netEC[v_num].modules():
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
            self.netF[v_num].train()
            self.netB[v_num].train()
            self.netE[v_num].train()
            #self.netC[v_num].train()
        elif mode=='test':
            self.netF[v_num].eval()
            self.netB[v_num].eval() 
            self.netC[v_num].eval() 
        f=self.netF[v_num](input)
        feature=self.netB[v_num](f)
        evidence=self.netEC[v_num](self.netE[v_num](f))

        out=self.netC[v_num](feature)
        
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
        self.alpha_a= self.DS_Combin(self.alpha,idx)
        self.evidence_a = self.alpha_a - 1
        return self.evi,self.out,self.features

 

    def integrated_forward(self,input,mode):
        #loss = 0
        self.out,self.features=self.forward(input,mode,input_mode='x')
        fused_output=self.out[0]*1
        self.fused_feature=self.features[0]*1
        for i in range(1,self.source):
            #pdb.set_trace()
            self.fused_feature=torch.cat((self.fused_feature,self.features[i]),1)
            fused_output=torch.cat((fused_output,self.out[i]),1)
        self.fused_output=self.bf_layer(fused_output)

        return self.fused_output,self.fused_feature

    def obtain_weight(self,weights):
        self.weights=weights
  
    def loss_model_ce(self,y_fused,epoch=1,c=65):
        loss = 0
      
        loss +=  nn.CrossEntropyLoss()(self.model_out,y_fused)
        loss = torch.mean(loss)
        return  loss
    
    def loss_model_kl(self):
        loss = 0
        prob=((self.alpha_a)/torch.sum(self.alpha_a,1,True)).detach()
        a=torch.max(prob,1)
        thre=torch.relu(a-0.8)
        loss +=  nn.KLDivLoss(reduce='none')(self.model_out.softmax(1).log(),prob)*thre
        loss = torch.mean(loss)
        return  loss
    def evi_klloss(self,idx):
        loss = 0
        sampled_labels=self.sample_label(self.evidence_a,3)
        thre=nn.Sigmoid()(torch.sum(self.evidence_a,1,True)-torch.sum(self.evi[idx],1,True)).to(torch.float32).cuda().detach()
        for i in range(3):
            label=sampled_labels[i].cuda().detach().to(torch.float32)
            loss+=1/3*(nn.KLDivLoss()(nn.Softmax(dim=1)(self.out[idx]).log(),label)*thre)
 
        loss = torch.mean(loss)
        return loss

    def loss_div(self):
        
        loss=0
        for v_num in range(self.source):
            for q_num in range(v_num,self.source):
                loss += nn.MSELoss()(self.features[v_num],self.features[q_num])
        return loss

    def loss_sub_mix_ce(self,y_list,v_num):
        loss = 0
        
        loss +=   CrossEntropy1()(self.out[v_num],y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss
    def loss_fuse_mix_ce(self,y_list):
        loss = 0
        
        loss +=   CrossEntropy1()(self.fused_output,y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss

    def loss_fuse_mix_ce(self,y_list):
        loss = 0
        
        loss += CrossEntropy1()(self.fused_output,y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss
    def loss_model_mix_ce(self,y_list):
        loss = 0
        for v_num in range(self.source):
            loss += CrossEntropy1()(self.model_out,y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss
    
    def loss_sub_ce(self,y_list,v_num):
        loss = 0
        
        loss +=   nn.CrossEntropyLoss()(self.out[v_num],y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss

    def loss_sub_ce_all_sub(self,y_list):
        loss = 0
        for v_num in self.source:
            loss +=   nn.CrossEntropyLoss()(self.out[v_num],y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss


    def loss_sub_ce_unc(self,y_list,um,v_num):
        loss = 0
        um=um.unsqueeze(1)
        
        loss +=  nn.CrossEntropyLoss()(self.out[v_num],y_list)+ torch.mean((um)*CrossEntropy2(reduction=False)(self.out[v_num],y_list))#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss
   
    def loss_sub_ce_sub_train(self,y_list,v_num):
        loss = 0
      
        loss += nn.CrossEntropyLoss()(self.out[v_num],y_list[v_num])#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss
    def loss_sub_ce_sub_train2(self,y_fused,v_num):
        loss = 0
      
        loss += nn.CrossEntropyLoss()(self.out[v_num],y_fused)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss
   
    def loss_sub_kl(self,prob,ud,um,k=5,cof=0.5,v_num=0):
        loss = 0
        ud=ud.unsqueeze(1)
        um=um.unsqueeze(1)
        prob=prob.float()

        values, indices=prob.topk(k,dim=1, largest=True, sorted=True)
        top={}
        target={}
        label={}
        for i in range(k):
            top[i]=indices[:,i].detach()
            target[i] = torch.zeros(self.fused_output.size()).scatter_(1, top[i].unsqueeze(1).cpu(), 1).detach().cuda()
        for i in range(0,k):
            label[i]=target[i]#cof*(ud)*target[i]/(k-1)
            label[i]=label[i].detach()
        #label=target[0]#*(1-cof*(ud))
        loss = (1-2*um)*CrossEntropy1(reduction=False)(self.out[v_num],label[0])
        for i in range(0,k):
            loss += (k-i)/k*ud*um*CrossEntropy1(reduction=False)(self.out[v_num],label[i])
        loss = torch.mean(loss)
        return  loss

    


    def loss_fuse_kl(self,prob,ud,um,k=5,cof=0.5):
        # loss = 0
        ud=ud.unsqueeze(1)
        um=um.unsqueeze(1)
        prob=prob.float()

        values, indices=prob.topk(k,dim=1, largest=True, sorted=True)
        
        top={}
        target={}
        label={}
        for i in range(k):
            top[i]=indices[:,i].detach()
            target[i] = torch.zeros(self.fused_output.size()).scatter_(1, top[i].unsqueeze(1).cpu(), 1).detach().cuda()
        for i in range(0,k):
            label[i]=target[i]#cof*(ud)*target[i]/(k-1)
            label[i]=label[i].detach()
        #label=target[0]#*(1-cof*(ud))
        loss = (1-(um+1-ud))*CrossEntropy1(reduction=False)(self.fused_output,label[0])
        for i in range(0,k):
            loss += (k-i)/k*ud*um*CrossEntropy1(reduction=False)(self.fused_output,label[i])
            #+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        loss = torch.mean(loss)
        return  loss

    


    def loss_fuse_ce(self,y_list):
        loss = 0
        
        loss += nn.CrossEntropyLoss()(self.fused_output,y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
        #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
        #loss = torch.mean(loss)
        return  loss


    # def loss_unc_ce(self,y_list):
    #     loss = 0
        
    #     loss += nn.CrossEntropyLoss()(self.fused_output[:-2],y_list)#+self.weight[v_num]*nn.CrossEntropyLoss()(self.prob[v_num],y_list[v_num])#ce_loss( y_fused, self.alpha[v_num],c)
    #     #loss +=  nn.CrossEntropyLoss()(self.out,y_fused)
    #     #loss = torch.mean(loss)
    #     return  loss



    def loss_model_total_entropy(self):
        loss = 0
        loss+= total_entropy_loss( nn.Softmax(dim=1)(self.model_out), self.classes)
        loss = torch.mean(loss)
        return  loss
    def loss_model_entropy(self):
        loss = 0
        loss += entropy_loss( nn.Softmax(dim=1)(self.model_out))
        loss = torch.mean(loss)
        return  loss

    def loss_kd(self):
        loss = 0
        
        target=self.output_a.detach()
        loss +=(nn.KLDivLoss()(self.model_out.softmax(1).log(),target))
        loss = torch.mean(loss)
        return  torch.mean(loss)
 

    def loss_model_entropy(self):
        loss = 0
        loss += entropy_loss( nn.Softmax(dim=1)(self.model_out))
        loss = torch.mean(loss)
        return  loss

    def loss_fuse_total_entropy(self):
        loss = 0
        loss+= total_entropy_loss( nn.Softmax(dim=1)(self.fused_output), self.classes)
        loss = torch.mean(loss)
        return  loss
        
    def loss_fuse_entropy_um(self,um):
        loss = 0
        um=um.unsqueeze(1)
        loss += (1-um)*(entropy_loss(nn.Softmax(dim=1)(self.fused_output)))
        loss = torch.mean(loss)
        return  loss

    def loss_fuse_entropy(self):
        loss = 0
        loss += entropy_loss(nn.Softmax(dim=1)(self.fused_output))
        loss = torch.mean(loss)
        return  loss
   
    def loss_sub_total_entropy(self,v_num):
        loss = 0
       
        loss +=  total_entropy_loss(nn.Softmax(dim=1)(self.out[v_num]),self.classes)
        
        loss = torch.mean(loss)
        return  loss
    def loss_sub_entropy(self,v_num):
        loss = 0
        loss +=   entropy_loss( nn.Softmax(dim=1)(self.out[v_num]), self.classes)
        loss = torch.mean(loss)
        return  loss
    def loss_sub_total_entropy_all_sub(self):
        loss = 0
        for v_num in self.source:
            loss +=  total_entropy_loss(nn.Softmax(dim=1)(self.out[v_num]),self.classes)
        
        loss = torch.mean(loss)
        return  loss
    def loss_sub_entropy_all_sub(self):
        loss = 0
        for v_num in self.source:
            loss +=   entropy_loss( nn.Softmax(dim=1)(self.out[v_num]), self.classes)
        loss = torch.mean(loss)
        return  loss
    
    def loss_sub_total_entropy_sub_train(self,v_num):
        loss = 0
       
        loss += total_entropy_loss(nn.Softmax(dim=1)(self.out[v_num]),self.classes)
        
        loss = torch.mean(loss)
        return  loss
    def loss_sub_entropy_sub_train(self,v_num):
        loss = 0
        
        loss += entropy_loss( nn.Softmax(dim=1)(self.out[v_num]), self.classes)
        loss = torch.mean(loss)
        return  loss
    