from stringprep import c22_specials
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

class CrossEntropy1(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=65, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropy1, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class CrossEntropy2(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=65, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropy2, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        #self.logsoftmax = #nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs =torch.log(inputs) #self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class CrossEntropy21(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=65, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropy21, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        #self.logsoftmax = #nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs =torch.log(inputs) #self.logsoftmax(inputs)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

class CrossEntropy3(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=65, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropy3, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        #self.logsoftmax = #nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs =torch.log(inputs) #self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def KL_KD(alpha, beta):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss_no_onehot(label, alpha, c=65, global_step=1, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    #label = F.one_hot(label, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # annealing_coef = min(1, global_step / annealing_step)
    # alp = E * (1 - label) + 1
    B =  KL(alpha, c)
    return torch.mean((A))


def ce_loss(label, alpha, c, global_step=1, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(label, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef =global_step / annealing_step
    alp = E * (1 - label) + 1
    B =  KL(alp, c)
    return torch.mean((A+B))

def ce_loss2(label, alpha, c, global_step=1, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(label, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef =global_step / annealing_step
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A+B))

    
def ce_loss3(label, alpha, c, global_step=1, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(label, num_classes=c)
    
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    #pred = F.one_hot(pred, num_classes=c)
    #annealing_coef = global_step / annealing_step
    #alp = E * (1 - pred) + 1
    #B = annealing_coef * KL(alp, c)
    return torch.mean((A))

def ce_loss_smooth(label, alpha, c=65, global_step=1, annealing_step=1,epsilon=0.1):
    
    targets = torch.zeros(alpha.size()).scatter_(1, label.unsqueeze(1).cpu(), 1)

    targets = (1 - epsilon) * targets +epsilon / c
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = targets.cuda()#F.one_hot(label, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = 1#min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A))#+0.1*B))

def ce_loss_log(label, alpha, c, global_step=1, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(label, num_classes=c)
    A = torch.sum(label * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

    # annealing_coef = min(1, global_step / annealing_step)
    # alp = E * (1 - label) + 1
    # B = annealing_coef * KL(alp, c)
    return torch.mean((A))




def metric_evidence(alpha, c):
    S = torch.sum(alpha, dim=1)
    E = alpha - 1
    unc=c/S
   
    #max_alpha=S-torch.max(alpha,1)[0]
    
    loss= torch.mean(unc) 
    return loss

def entropy_loss(p,c=65):
    # S = torch.sum(alpha, dim=1, keepdim=True)
    # p=alpha/S
    #p=nn.Softmax(dim=1)(alpha)
    epsilon = 1e-5
    entropy = -p * torch.log(p + epsilon)
    #entropy*= (c/S)
    #pdb.set_trace()
    loss = torch.sum(entropy, dim=1)
    #loss=torch.mean(entropy) 
    return loss


def total_entropy_loss( p,c=65):
    #p=p*(S)
    msoftmax = p.mean(dim=0) 
    epsilon = 1e-5
    loss = torch.sum(-msoftmax * torch.log(msoftmax +epsilon))
    # S = torch.sum(alpha)

    #loss=torch.mean(entropy) 
    return loss


def entropy_loss1( alpha,c):
    # S = torch.sum(alpha, dim=1, keepdim=True)
    # E = alpha - 1
    # p=alpha/S
    # #label = F.one_hot(label, num_classes=c)
    # loss = torch.sum(p* (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    S = torch.sum(alpha, dim=1, keepdim=True)
    A=torch.lgamma(alpha)
    A=torch.sum(A, dim=1)
    B=torch.lgamma(S)
    C=(S-c)*torch.digamma(S)
    D=torch.sum( (alpha-1)*torch.digamma(alpha), dim=1, keepdim=True)
    loss=(A-B+C-D)/S
 
    loss=torch.mean(loss) 
    return loss

def total_entropy_loss1( alpha,c):
  
    S = torch.sum(alpha)
    A=torch.lgamma(alpha)
    A=torch.sum(A)
    B=torch.lgamma(S)
    C=(S-c)*torch.digamma(S)
    D=torch.sum( (alpha-1)*torch.digamma(alpha))
    loss=torch.log(A-B)+C-D
    loss=torch.mean(loss) 
    return loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def KLConsistencyLoss(output, pred_label, args, temperature=2):
    """
    Class-Relation-Aware Consistency Loss
    Args:
        output: n x b x k (source num x batch size x class num)
        pred_label:  b x 1
        args:   argments
    """
    eps = 1e-16
    KL_loss = 0

    label_id = pred_label.cpu().numpy()
    label_id = np.unique(label_id)

    for cls in range(args.class_num):
        if cls in label_id:
            prob_cls_all = torch.ones(len(args.src), args.class_num)

            for i in range(len(args.src)):
                mask_cls =  pred_label.cpu() == cls
                mask_cls_ex = torch.repeat_interleave(mask_cls.unsqueeze(1), args.class_num, dim=1)

                logits_cls = torch.sum(output[i] * mask_cls_ex.float(), dim=0)
                cls_num = torch.sum(mask_cls)
                logits_cls_acti = logits_cls * 1.0 / (cls_num + eps)
                prob_cls = torch.softmax(logits_cls_acti, dim=0)
                prob_cls = torch.clamp(prob_cls, 1e-8, 1.0)

                prob_cls_all[i] = prob_cls


            for m in range(len(args.src)):
                for n in range(len(args.src)):
                    KL_div = torch.sum(prob_cls_all[m] * torch.log(prob_cls_all[m] / prob_cls_all[n])) + \
                              torch.sum(prob_cls_all[n] * torch.log(prob_cls_all[n] / prob_cls_all[m]))
                    KL_loss += KL_div / 2

    KL_loss = KL_loss / (args.class_num * len(args.src))

    return KL_loss



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss