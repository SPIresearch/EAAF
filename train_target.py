import argparse
from concurrent.futures import thread

import os, sys


import os.path as osp

from torch.cuda import amp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import argmax, linalg as LA
from torchvision import transforms
import network as network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx,Listset,Listset3,ImageList_aug
import random, pdb, math, copy
from tqdm import tqdm
from randaug import RandAugmentMC
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix





def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def positive_aug(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        #transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])




def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,pin_memory=True,
                                        drop_last=False)
    dsets['target_'] = ImageList_idx(txt_tar,  transform=image_train(),transform1=positive_aug())
    dset_loaders['target_'] = DataLoader(dsets['target_'], batch_size=train_bs, shuffle=True,pin_memory=True,
                                    drop_last=False)
 
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 10, shuffle=False,pin_memory=True,
                                      drop_last=False)

    return dset_loaders,dsets


def sigma3(ten_sor,k=0.5):
    u=torch.mean(ten_sor)
    v=torch.std(ten_sor)
    thread_=u+k*v
    l_id=torch.nonzero(ten_sor<thread_).squeeze().cpu().numpy()
    h_id=torch.nonzero(ten_sor>thread_).squeeze().cpu().numpy()
    return l_id,h_id


def build_data_source(label):
    data_source=[]
    for i in range(len(label)):
        data_source.append([i,label[i].item()])

    return data_source
def mixup(input,label):
    alpha = 0.75
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(input.size()[0]).cuda()
    mixed_input = lam *input + (1 - lam) * input[index, :]
    label = (lam * label + (1 - lam) * label[index, :]).detach()
    return mixed_input,label

def get_list(a,b):
    
    if a.shape != () and b.shape !=  ():
        tmp = [val for val in a if val in b] 
    else: 
        tmp=[]
    return tmp

def train_target_primary(args,dset_loaders,netF,netB,netC,netE,netEC,primary_idx,unc_list,evi_list,out_list,pse_list,fea_list):
    def DS_Combin_two(alpha1, alpha2,):
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
            u[v] = args.class_num/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, args.class_num, 1), b[1].view(-1, 1, args.class_num))
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
        S_a = args.class_num / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        
        return alpha_a
    def initial(args):
        param_group = []
        param_group_c=[]
        for k, v in netF.named_parameters():
            if args.lr_decay1 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False
        for k, v in netB.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False
        for k, v in netE.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False
        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)
        for k, v in netC.named_parameters():
            v.requires_grad = False
            #param_group_c += [{"params": v, "lr": args.lr * 1}] 
        for k, v in netEC.named_parameters():
            param_group_c += [{"params": v, "lr": args.lr * 1}]  

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        optimizer_c = optim.SGD(param_group_c)
        optimizer_c = op_copy(optimizer_c)
        return netF,netB,netC,netE,netEC,optimizer,optimizer_c

    netF,netB,netC,netE,netEC,optimizer,optimizer_c=initial(args)
    max_iter =len(dset_loaders["target"])
   
    iter_num = 0

    
    netF.eval()
    netB.eval()
    netC.eval()
    num_sample = len(dset_loaders["target"].dataset)
    fea_bank=dict()
  
    fea_bank = torch.randn(num_sample, 256*3)
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    
   
    fea_list_norm=fea_list
    for k in fea_list_norm.keys():
        fea_list_norm[k]=F.normalize(fea_list_norm[k])
    for i in range (3):
        if i==primary_idx:
            with torch.no_grad():
                iter_test = iter(dset_loaders["target"])
                for i in range(len(dset_loaders["target"])):
                    data = iter_test.next()
                    inputs = data[0]
                    indx = data[-1]
                    # labels = data[1]
                    inputs = inputs.cuda()
                    fea=netF(inputs)
                    fea = netB(fea)
                    fea_norm = F.normalize(fea)
                    for i in range(3):
                        if i==primary_idx:
                            continue
                        fea_norm=torch.cat([fea_norm,fea_list_norm[i][indx].cuda()],1)
                    outputs = netC(fea)
                    outputs = nn.Softmax(-1)(outputs)

                    fea_bank[indx] = fea_norm.detach().clone().cpu()
                    score_bank[indx] = outputs.detach().clone()  # .cpu()
    epoch=0
    
    fuse_epoch=args.max_epoch
    while epoch < fuse_epoch:
        epoch+=1
        iter_num=0
        lr_scheduler(optimizer, iter_num=epoch, max_iter=fuse_epoch)
        netF.eval()
        netB.eval()
        netE.eval()
        netC.eval()
        netEC.eval()
       
        mem_label,mem_sec_label,all_evidence,all_label = obtain_label(dset_loaders['test'], netF, netB, netC,netE, netEC,primary_idx,out_list, fea_list,evi_list,args)
        #pdb.set_trace()
        unc_list[primary_idx]=args.class_num/(torch.max(all_evidence+1,1)[0])
       
        mem_label = torch.from_numpy(mem_label)

        pse_list[primary_idx]=mem_label
        netF.train()
        netB.train()
        netE.train()
        
        print('original acc:',test_acc(pse_list[primary_idx],list(range(len(unc_list[0]))),all_label))
        





        

        if epoch<=1:
            source_pse=torch.zeros(mem_label.shape[0],3)
            source_unc=torch.zeros(mem_label.shape[0],3)
            for i in range(3):
                if i!=primary_idx:
                    source_pse[:,i]=pse_list[i]
                    source_unc[:,i]=unc_list[i]
                    print('test acc:',test_acc(source_pse[:,i],list(range(len(unc_list[0]))),all_label))
                else:
                    source_pse[:,i]=pse_list[i]
                    source_unc[:,i]=unc_list[i]*0.6
                    print('test primary_idx acc:',test_acc(source_pse[:,i],list(range(len(unc_list[0]))),all_label))
        else:       
            source_pse[:,primary_idx]=pse_list[primary_idx]
            source_unc[:,primary_idx]=unc_list[primary_idx]*0.6
        sorts=torch.argmin(source_unc,1)
        
        PLE=source_pse[:,sorts]


        mem_sec_label = torch.from_numpy(mem_sec_label)
        threshold=torch.ones(args.class_num)
        threshold_unc=torch.ones(args.class_num)

        local_inconsistency_mask=torch.zeros(mem_label.shape[0])
        pse_smooth_mask=torch.zeros(mem_label.shape[0])
        all_alpha=all_evidence+1
        
        ELI=torch.log(1+(torch.sum(all_alpha)-torch.max(all_alpha,1)[0])/(torch.max(all_alpha,1)[0]))
        uncertainty=args.class_num/(torch.max(all_alpha,1)[0]+args.class_num)
        for i in range(args.class_num):
            eli_class_i=ELI[mem_label==i]
            threshold[i]=torch.mean(eli_class_i)+2*torch.std(eli_class_i)

        for i in range(args.class_num):
            unc_class_i=uncertainty[mem_label==i]
            threshold_unc[i]=torch.mean(unc_class_i)+2*torch.std(unc_class_i)

        for i in range(mem_label.shape[0]):
            p=mem_label[i]
            if ELI[i]>threshold[p]:
                local_inconsistency_mask[i]=1
            else:
                local_inconsistency_mask[i]=0

        for i in range(mem_label.shape[0]):
            p=mem_label[i]
            if uncertainty[i]>threshold_unc[p]:
                pse_smooth_mask[i]=1
            else:
                pse_smooth_mask[i]=0
        

        
        while iter_num < max_iter:
            try:
                inputs_test, label, tar_idx = iter_test.next()
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, label, tar_idx = iter_test.next()

            if inputs_test.size(0) == 1:
                continue
            iter_num += 1
            agg_loss = torch.tensor(0.0).cuda()
            inputs_test = inputs_test.cuda()
            fea=netF(inputs_test)
            evidence = netEC(netE(fea))

            alpha = evidence+1
            S=torch.sum(alpha,1,keepdim=True)
            features_test = netB(fea)
            outputs_test = netC(features_test)
            #pred=mem_label[tar_idx].cuda()
            #label=label.cuda()
            with torch.no_grad():
                f=netF(inputs_test)
                f1 = netB(f)
                o1 = netC(f1).cpu()
                out_list[primary_idx][tar_idx] =o1.cpu()
                primary_evidence = netEC(netE(f)).detach()
            agg_loss_mul= torch.tensor(0.0).cuda()
            agg_loss_evi= torch.tensor(0.0).cuda()
            for i in range(3):
                if i==primary_idx:
                    pred_u=PLE[i][tar_idx].long().cuda()
                    weak_evidence=evi_list[i][tar_idx].cuda()
                    fused_alpha=DS_Combin_two(primary_evidence+1,weak_evidence+1)
                    
                    combined_S=torch.sum(fused_alpha,1,keepdim=True)
                    #pdb.set_trace()
                    pse_loss_ele=loss.ce_loss2(pred_u,alpha,args.class_num,iter_num+(epoch-1)*max_iter,( fuse_epoch-1)*max_iter)+torch.mean( nn.CrossEntropyLoss(reduction='none')(outputs_test,pred_u))
                if i!=primary_idx:
                    mask=(unc_list[i][tar_idx]<unc_list[primary_idx][tar_idx]+0).unsqueeze(1).cuda()
                    pred_u=out_list[i][tar_idx]+out_list[primary_idx][tar_idx]
                    pred_u=nn.Softmax(1)(pred_u)
                    _,pred_u=torch.max(pred_u,1)
                    pred_u=pred_u.cuda()
                    weak_evidence=evi_list[i][tar_idx].cuda()
                    fused_alpha=DS_Combin_two(primary_evidence+1,weak_evidence+1)
                    
                    combined_S=torch.sum(fused_alpha,1,keepdim=True)
                    #pdb.set_trace()
                    agg_loss_evi+=1/2*torch.mean( nn.KLDivLoss(reduction='none')((fused_alpha/combined_S).log(),alpha/S)*mask)
                    agg_loss_mul+=1/2*torch.mean( nn.CrossEntropyLoss(reduction='none')(outputs_test,pred_u)*mask)

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))

                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))


                    entropy_loss -=gentropy_loss
                im_loss = entropy_loss * args.ent_par
            agg_loss= 0.1*pse_loss_ele+agg_loss_evi+agg_loss_mul+im_loss
            #agg_loss+=nn.CrossEntropyLoss()(outputs_test,pred)
            optimizer.zero_grad()
            agg_loss.backward()
            optimizer.step()
            if epoch<=5 and args.t==0:
                continue
            try:
                [inputs_test,inputs_test1], _, tar_idx = iter_test_.next()
            except:
                iter_test_ = iter(dset_loaders["target_"])
                [inputs_test,inputs_test1], _, tar_idx = iter_test_.next()

            if inputs_test.size(0) == 1:
                continue

            inputs_test = inputs_test.cuda()
            inputs_test1 = inputs_test.cuda()

            pred_u1=mem_label[tar_idx]
            targets_1 = torch.zeros((pred_u1.size(0),65)).scatter_(1, pred_u1.unsqueeze(1).cpu(), 1)
            pred_u2=mem_sec_label[tar_idx]
            targets_2 = torch.zeros((pred_u2.size(0),65)).scatter_(1, pred_u2.unsqueeze(1).cpu(), 1)
            unc=uncertainty[tar_idx].unsqueeze(1)
            targets=targets_1*(1-0.5*unc)+targets_2*(0.5*unc)
            smoothed=(pse_smooth_mask[tar_idx]==1)
            #print("len smooth",sum(smoothed),'acc',test_acc(pred_u1,smoothed,all_label[tar_idx]))
            targets_1[smoothed,:]=targets[smoothed,:]
            targets_smooth=targets_1.cuda()

            all_inputs = torch.cat([inputs_test, inputs_test1], dim=0)
            #all_targets = torch.cat([targets, targets], dim=0).cuda()

            
            lc_mask=local_inconsistency_mask[tar_idx].cuda()
            features_test = netB(netF(all_inputs))
            outputs_test = netC(features_test)
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            # output_re = softmax_out.unsqueeze(1)

            with torch.no_grad():
                output_f_norm = F.normalize(features_test[:features_test.shape[0]//2])
                output_f_ = output_f_norm.cpu().detach().clone()

                
                fea_bank[tar_idx][:,:256] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = softmax_out[:features_test.shape[0]//2].detach().clone()

                distance = fea_bank[tar_idx] @ fea_bank.T /3
                dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                dis_near = dis_near[:, 1:]
                score_near = score_bank[idx_near]  # batch x K x C

            # nn
            softmax_out_un = softmax_out[:features_test.shape[0]//2].unsqueeze(1).expand(
                -1, args.K, -1
            )  # batch x K x C
            #pdb.set_trace()
            local_loss_sec = torch.mean((F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)) # Equal to - dot product
            local_loss_risk= -1*torch.mean(lc_mask*((dis_near.cuda()*(F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1))).sum(1)))
            local_loss=local_loss_sec+local_loss_risk
            local_loss+=torch.mean(F.kl_div(softmax_out[:features_test.shape[0]//2],softmax_out[features_test.shape[0]//2:], reduction="none"))
            pse_loss_smoooth=loss.CrossEntropy1(args.class_num)(outputs_test[:features_test.shape[0]//2],targets_smooth)
            
            mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy_soft = softmax_out[:features_test.shape[0]//2].T  # .detach().clone()#

            dot_neg = softmax_out[:features_test.shape[0]//2] @ copy_soft  # batch x batch

            dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
            neg_pred = torch.mean(dot_neg)
            local_loss += neg_pred 
            adapt_loss=0.1*pse_loss_smoooth+local_loss
            optimizer.zero_grad()
            #optimizer_c.zero_grad()
            adapt_loss.backward()
            optimizer.step()

    
        netF.eval()
        netB.eval()
        netE.eval()
        netEC.eval()
        if args.dset=='VISDA-C':
            acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, epoch, fuse_epoch, acc_s_te) + '\n' + acc_list
        else:
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, epoch,  fuse_epoch, acc_s_te)

        
        print(log_str+'\n')
        netF.train()
        netB.train()
        netE.train()
        netEC.train()
            
        #optimizer_c.step()
            

        
        

        # while iter_num < max_iter:
        #     try:
        #         inputs_test, label, tar_idx = iter_test.next()
        #     except:
        #         iter_test = iter(dset_loaders["target"])
        #         inputs_test, label, tar_idx = iter_test.next()

        #     if inputs_test.size(0) == 1:
        #         continue
        #     iter_num += 1
        #     lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        #     lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
        #     inputs_test = inputs_test.cuda()
        #     features_test = netB(netF(inputs_test))
        #     outputs_test = netC(features_test)
        #     if args.cls_par > 0:
        #         pred = mem_label[tar_idx]
        #         agg_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        #         agg_loss *= args.cls_par
        #         if iter_num < interval_iter and args.dset == "VISDA-C":
        #             agg_loss *= 0
        #     else:
        #         agg_loss = torch.tensor(0.0).cuda()

        #     if args.ent:
        #         softmax_out = nn.Softmax(dim=1)(outputs_test)
        #         entropy_loss = torch.mean(loss.Entropy(softmax_out))

        #         if args.gent:
        #             msoftmax = softmax_out.mean(dim=0)
        #             gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))


        #             entropy_loss -=gentropy_loss
        #         im_loss = entropy_loss * args.ent_par
        #         agg_loss += im_loss

        #     optimizer.zero_grad()
        #     agg_loss.backward()
        #     optimizer.step()
            
            
        

    

    # if args.issave:   
    #     torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F"  + ".pt"))
    #     torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B"  + ".pt"))
    #     torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C"  + ".pt"))
        
    return netF, netB, netC


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def obtain_label2(loader, netF, netB, netC,netE, netEC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = (netF(inputs))
            #pdb.set_trace()
            evidence = netEC(netE(fea))
            feas=netB(fea)
            
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_evidence= evidence.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_evidence = torch.cat((all_evidence, evidence.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
  
        
    
    print(log_str+'\n')

    return pred_label.astype('int'),all_evidence,all_label

    

def obtain_label(loader, netF, netB, netC,netE, netEC,primary_idx,all_o_list, all_f_list,all_e_list,args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = (netF(inputs))
            #pdb.set_trace()
            evidence = netEC(netE(fea))
            feas=netB(fea)
            
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_evidence= evidence.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_evidence = torch.cat((all_evidence, evidence.float().cpu()), 0)
    #for i in range(3):
    all_o_list[primary_idx]=all_output
    all_f_list[primary_idx]=all_fea
    all_e_list[primary_idx]=all_evidence
    #all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    #all_output=#all_o_list[0]*weights[0]
    #for i in range(1,3):
    #    all_output+=all_o_list[i]*weights[i]

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) 
    if primary_idx==0:
        all_fea=all_f_list[0]
    else:
        all_fea=all_f_list[0]*0.5
    for i in range(1,3):
        if i==primary_idx:
            all_fea=torch.cat((all_fea,all_f_list[i]),1)
        else:
            all_fea=torch.cat((all_fea,all_f_list[i]*0.5),1)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
   

    all_fea = all_fea.float().cpu().numpy()
    all_fea=all_fea
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)

        pred_label = np.argsort(dd)[:,0]
        pred_label2=np.argsort(dd)[:,1]
        pred_label = labelset[pred_label]
        
        pred_label2 = labelset[pred_label2]
    acc = np.sum(pred_label2 == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy2 = {:.2f}%'.format( acc * 100)
    print(log_str+'\n')
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
   
        
    
    print(log_str+'\n')

    return pred_label.astype('int'),pred_label2.astype('int'),all_evidence,all_label

    



def initial(net,args):
    param_group = []
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        #print(modelpath)
        net.netF[i].load_state_dict(torch.load(modelpath))
        net.netF[i].eval()
        for k, v in net.netF[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        #print(modelpath)
        net.netB[i].load_state_dict(torch.load(modelpath))
        net.netB[i].eval()

        for k, v in net.netB[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        #print(modelpath)
        net.netC[i].load_state_dict(torch.load(modelpath))
        net.netC[i].eval()
        for k, v in net.netC[i].named_parameters():
            v.requires_grad = False
        modelpath = args.output_dir_src[i] + '/source_E.pt'
        #print(modelpath)
        net.netE[i].load_state_dict(torch.load(modelpath))
        net.netE[i].eval()
        for k, v in net.netE[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        modelpath = args.output_dir_src[i] + '/source_EC.pt'
        #print(modelpath)
        net.netEC[i].load_state_dict(torch.load(modelpath))
        net.netEC[i].eval()    
        for k, v in net.netEC[i].named_parameters():
            v.requires_grad = False
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return net,optimizer   

def train_target(args):
    dset_loaders,dsets = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF_list = [network.ResBase(res_name=args.net).cuda() for i in range(len(args.src))]
      
    elif args.net[0:3] == 'vgg':
        netF_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
                                    
    netC_list = [network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    
    netE_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netEC_list = [network.evidence_classifier(type='linear', class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
   
    
    #netQ = network.MyLinear(256,256).cuda()
    net=network.ME(netF_list,netB_list,netC_list,netE_list,netEC_list,args.class_num,len(args.src), args.max_epoch * len(dset_loaders["target"]) // args.interval)

   
    


    net,optimizer=initial(net,args)
    
    out_list,evi_list,pse_list,all_f_list,all_label=pre_infer(dset_loaders['test'],net)
    pred=dict()
    consistency=[]
    unc=dict()
    prev=dict()
    for i in range(net.source):
    
        prev[i], pred[i]= torch.max(out_list[i], 1)
        unc[i]=args.class_num/(torch.max(evi_list[i]+1,1)[0])
        consistency.append(torch.sum(torch.max(evi_list[i],1)[0] ))
        accuracy = torch.sum(torch.squeeze(pred[i]).float() == all_label).item() / float(all_label.size()[0])
        print(accuracy)
    max_model=np.argmax(np.array(consistency))
    print('max_model',max_model)
    net=train_target_primary(args,dset_loaders,net.netF[max_model],net.netB[max_model],net.netC[max_model],net.netE[max_model],net.netEC[max_model],max_model,unc,evi_list,out_list,pse_list,all_f_list)
    
                        



def sub_pse_label(loader, net):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _,out_list,f_list=net.forward(inputs,'test',-1)

            if start_test:

                all_label = labels.float()
                all_f_list=dict()
                all_o_list=dict()
               
                for i in range(net.source):
                    all_f_list[i]=f_list[i].float().cpu()
                    all_o_list[i]=out_list[i].float().cpu()
                    
                start_test = False

            else:
               
                all_label = torch.cat((all_label, labels.float()), 0)
              
                for i in range(net.source):
                    all_o_list[i] = torch.cat((all_o_list[i], out_list[i].float().cpu()), 0)

                    all_f_list[i] = torch.cat((all_f_list[i], f_list[i].float().cpu()), 0)
   
    
    pred=dict()
    pred_label_list=dict()
    dd=dict()
    initc=dict()
    for i in range(net.source):
        _, predict = torch.max(all_o_list[i], 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        print('virtual: teacher model',i,':',accuracy)
 

    
    for i in range(net.source):
        _, pred[i]= torch.max(all_o_list[i], 1)
        pred_label_list[i],dd[i],initc[i]=cluster(nn.Softmax(1)(all_o_list[i]),all_f_list[i],all_label)
      
        acc = np.sum(pred_label_list[i] == all_label.float().numpy()) /(all_label.shape[0])
        # #pred_label_list[i]=pred_label_list[i].detach()
        # #initc[i]=initc[i].detach()
        # all_f_list[i]=all_f_list[i].detach()
        # #pred_label_list[i]=torch.from_numpy(pred_label_list[i].astype('int'))
        log_str = 'virtual: Accuracy of model {} -> {:.2f}%'.format(i,acc * 100)
        print(log_str)

    
    return pred_label_list,all_o_list,all_f_list,all_label



def pre_infer(loader, net):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]

            inputs = inputs.cuda()

            evi_list,out_list,f_list=net.forward(inputs,'test',-1)
            features=f_list[0]

            for i in range(1,net.source):
                features=torch.cat((features, f_list[i]), 1)

            for i in range(1,net.source):
                features=torch.cat((features, f_list[i]), 1)

            if start_test:
                
                all_features = features.float().cpu()
               
                all_label = labels.float()
                all_f_list=dict()
                all_o_list=dict()
                all_evi_list=dict()
                for i in range(net.source):
                    all_f_list[i]=f_list[i].float().cpu()
                    all_o_list[i]=out_list[i].float().cpu()
                    all_evi_list[i]=evi_list[i].float().cpu()

                start_test = False

            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
               
                all_label = torch.cat((all_label, labels.float()), 0)
               
                for i in range(net.source):
                    all_o_list[i] = torch.cat((all_o_list[i], out_list[i].float().cpu()), 0)

                    all_evi_list[i] = torch.cat((all_evi_list[i], evi_list[i].float().cpu()), 0)
                    all_f_list[i] = torch.cat((all_f_list[i], f_list[i].float().cpu()), 0)
    pred=dict()
    prev=dict()
    #consistency=[]
    for i in range(net.source):
       

        prev[i], pred[i]= torch.max(all_o_list[i], 1)
        #consistency.append(torch.sum(torch.max(all_evi_list[i],1)[0] ))
        accuracy = torch.sum(torch.squeeze(pred[i]).float() == all_label).item() / float(all_label.size()[0])
        print('model:',i,'acc:' ,accuracy,' evi:',torch.max(all_evi_list[i]))
        x=sigma3(torch.sum(all_evi_list[i],1),3)
        #x=sigma3(torch.max(all_evi_list[i],1)[0],3)
        print('test_evi',test_acc(pred[i],x[0],all_label),test_acc(pred[i],x[1],all_label))
    all_pse_list=dict()
    dd=dict()
    initc=dict()
    for i in range(net.source):
        _, pred[i]= torch.max(all_o_list[i], 1)
        all_pse_list[i],dd[i],initc[i]=cluster(nn.Softmax(1)(all_o_list[i]),all_f_list[i],all_label)
      
        acc = np.sum(all_pse_list[i] == all_label.float().numpy()) /(all_label.shape[0])
        # #pred_label_list[i]=pred_label_list[i].detach()
        # #initc[i]=initc[i].detach()
        # all_f_list[i]=all_f_list[i].detach()
        all_pse_list[i]=torch.from_numpy(all_pse_list[i].astype('int'))
        log_str = 'virtual: Accuracy of model {} -> {:.2f}%'.format(i,acc * 100)
        print(log_str)
        print('test_evi',test_acc(all_pse_list[i],x[0],all_label),test_acc(all_pse_list[i],x[1],all_label))
    
        all_f_list[i]=all_f_list[i].float().cpu()#.numpy()
    return  all_o_list,all_evi_list,all_pse_list,all_f_list,all_label

   
    
 
    
    
    

def test_acc(predict,hcl,all_label):
    hcl=torch.tensor(hcl)
   
    acc=torch.sum(torch.squeeze(predict[hcl]).cpu().float() == all_label[hcl]).item() / float(all_label[hcl].size()[0])
    return acc
    
def test_acc_from_prob(prob,hcl,all_label,k=0):
    hcl=torch.tensor(hcl)
    values, indices=prob.topk(k+1,dim=1, largest=True, sorted=True)
    predict=indices[:,k].detach()
    acc=torch.sum(torch.squeeze(predict[hcl]).cpu().float() == all_label[hcl]).item() / float(all_label[hcl].size()[0])
    return acc
   
  

def cluster(all_output,all_features,all_label):
    
    
    K = all_output.size(1)
    all_fea=all_features.cpu()
    
    # all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    # all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    aff = (all_output).float().cpu().numpy()
    
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
   
    # print(labelset)

    dd = cdist(all_fea, initc, args.distance)
    pred_label= dd.argmin(axis=1)
   
    
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd_fuse = cdist(all_features, initc, args.distance)

        pred_label = dd_fuse.argmin(axis=1)

    return pred_label,dd,initc


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
  


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAiDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0,
                        help="target")  ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office31', 'office-home', 'office-caltech','domainnet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.7)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--crc_par', type=float, default=1e-2)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--ema', type=float, default=0.6)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/v32')
    parser.add_argument('--output_src', type=str, default='ckps/source_mul_evi')
    parser.add_argument('--output_tar', type=str, default='ckps/target')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'domainnet':
        names = ['quickdraw','clipart', 'infograph', 'painting', 'sketch','real']
        args.class_num = 345
    for k in range(0,4):
        args.t=k
        args.src = []
        for i in range(len(names)):
            if i == args.t:
                continue
            else:
                args.src.append(names[i])

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        for i in range(len(names)):
            if i != args.t:
                continue

            folder = '/home/spi/peijiangbo/'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            print(args.t_dset_path)
       
        args.output_dir_tar = []
        args.output_dir_src= []
        args.name = names[args.t][0].upper()
        for i in range(len(args.src)):
            args.output_dir_tar.append(osp.join(args.output_tar, args.dset, args.src[i][0].upper()+names[args.t][0].upper()))
            args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i][0].upper()))
        print(args.output_dir_tar)
        print(args.output_dir_src)
        args.output_dir = osp.join(args.output, args.dset, names[args.t][0].upper())

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par) + '_' + str(args.crc_par)

        train_target(args)
