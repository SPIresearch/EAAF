import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import ce_loss,KL
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
  # if not alexnet:
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  # else:
  #   normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  # if not alexnet:
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  # else:
  #   normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def model_forward(mddn_F,mddn_C1,mddn_C2,mddn_E1,mddn_E2, X, y):
    fea=mddn_F(X)
    output=mddn_C2(mddn_C1(fea))
    #pred=(torch.argmax(output)).detach()
    evidence = mddn_E2(mddn_E1(fea))

    alpha = evidence+1
    #pred_y=output.detach().clone()
    #pdb.set_trace()
    #loss1 = ce_loss(y, alpha, classes, global_step, annealing_step)
    #loss1 = torch.mean(loss1)
    loss2=nn.CrossEntropyLoss()(output, y) #CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(output, y) 
    #pe=alpha/  torch.sum(alpha,1,keepdim=True)
    #p= nn.Softmax(dim=1)(output)
    #loss3=nn.KLDivLoss()(p.log(),pe)    
    loss=loss2#+loss1 
    return evidence,loss



def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, mddn_F, mddn_C1, mddn_C2, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = (mddn_C1(mddn_F(inputs)))
            outputs = mddn_C2(outputs)#simple_transform(outputs,1.3))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_source_step1(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        mddn_F = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        mddn_F = network.VGGBase(vgg_name=args.net).cuda()  

    mddn_C1 = network.feat_bottleneck(type=args.bn, feature_dim=mddn_F.in_features, bottleneck_dim=args.bottleneck).cuda()
    mddn_C2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    mddn_E1=network.feat_bottleneck(type=args.bn, feature_dim=mddn_F.in_features, bottleneck_dim=args.bottleneck).cuda()
    mddn_E2 = network.evidence_classifier(type='linear', class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    param_group = []
    learning_rate = args.lr

    for k, v in mddn_F.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in mddn_C1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in mddn_C2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}] 
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])//2
    interval_iter = max_iter
    iter_num = 0

    mddn_F.train()
    mddn_C1.train()
    mddn_C2.train()
    mddn_E2.eval()
    mddn_E1.eval()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        #outputs_source = mddn_C2(mddn_C1(mddn_F(inputs_source)))

        #classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        _,classifier_loss=model_forward(mddn_F,mddn_C1,mddn_C2,mddn_E1,mddn_E2,inputs_source,labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            mddn_F.eval()
            mddn_C1.eval()
            mddn_C2.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], mddn_F, mddn_C1, mddn_C2,False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            mddn_F.train()
            mddn_C1.train()
            mddn_C2.train()
                
  
    return mddn_F, mddn_C1, mddn_C2,mddn_E1,mddn_E2
def simple_transform(x, beta):
            x = 1/torch.pow(torch.log(1/x+1),beta)
            return x
def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        mddn_F = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        mddn_F = network.VGGBase(vgg_name=args.net).cuda()  

    mddn_C1 = network.feat_bottleneck(type=args.bn, feature_dim=mddn_F.in_features, bottleneck_dim=args.bottleneck).cuda()
    mddn_C2 = network.evidence_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    mddn_F.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    mddn_C1.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    mddn_C2.load_state_dict(torch.load(args.modelpath))
    mddn_F.eval()
    mddn_C1.eval()
    mddn_C2.eval()

    acc, _ = cal_acc_test(dset_loaders['test'], mddn_F, mddn_C1, mddn_C2, False,args)
    log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
def cal_acc_test(loader, mddn_F, mddn_C1, mddn_C2,flag,args):
   
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]

            inputs = inputs.cuda()
            
            features = (mddn_C1(mddn_F((inputs))))
            outputs = mddn_C2(features)#+1#simple_transform(outputs,1.3))
            
           

            if start_test:
                all_features = features.float().cpu()
                all_inputs = inputs.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)


    all_output = nn.Softmax(dim=1)(all_output)
    prob, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
  
    import scipy.io
    scipy.io.savemat(f'hsn_{str(args.s)}_{str(args.t)}_tsne.mat', {'ft':all_features.numpy(),'label':all_label.numpy(),'output':all_output.numpy()})
   
    all_fea=all_features

    #all_output = nn.Softmax(dim=1)(all_output)

    # alpha = all_output+1
    # S = torch.sum(alpha, dim=1, keepdim=True)
    # all_output=alpha/S

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    args.distance = 'cosine'
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>=0)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(0):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
  
        
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
   


    #writer.add_embedding(selected_feature, metadata=meta_label, label_img=selected_data)

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def train_source_step2(args,mddn_F, mddn_C1, mddn_C2,mddn_E1,mddn_E2):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        mddn_F = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        mddn_F = network.VGGBase(vgg_name=args.net).cuda()  

   
    param_group = []
    learning_rate = args.lr
    
    for k, v in mddn_F.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in mddn_C1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in mddn_C2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}] 
    for k, v in mddn_E1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}] 
    for k, v in mddn_E2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])//2
    interval_iter = max_iter#// 2
    iter_num = 0

    mddn_F.train()
    mddn_C1.train()
    mddn_C2.train()
    mddn_E2.train()
    mddn_E1.train()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        #outputs_source = mddn_C2(mddn_C1(mddn_F(inputs_source)))

        #classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        _,classifier_loss=model_forward(mddn_F,mddn_C1,mddn_C2,mddn_E1,mddn_E2,inputs_source,labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            mddn_F.eval()
            mddn_C1.eval()
            mddn_C2.eval()
            mddn_E1.eval()
            mddn_E2.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], mddn_F, mddn_C1, mddn_C2,False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_mddn_F = mddn_F.state_dict()
                best_mddn_C1 = mddn_C1.state_dict()
                best_mddn_C2 = mddn_C2.state_dict()
                best_mddn_E1 = mddn_E1.state_dict()
                best_mddn_E2 = mddn_E2.state_dict()
            mddn_F.train()
            mddn_C1.train()
            mddn_C2.train()
            mddn_E1.train()
            mddn_E2.train()
    torch.save(best_mddn_F, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_mddn_C1, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_mddn_C2, osp.join(args.output_dir_src, "source_C.pt"))
    torch.save(best_mddn_E1, osp.join(args.output_dir_src, "source_E.pt"))
    torch.save(best_mddn_E2, osp.join(args.output_dir_src, "source_EC.pt"))
    return mddn_F, mddn_C1, mddn_C2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAiDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--bn', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ckps/source_mul_evi2')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
   
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65 
   

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    for k in range(len(names)):
        args.s=k
        folder = '/home/spi/peijiangbo/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'     
     
        args.output_dir_src = osp.join(args.output, args.dset, names[args.s][0].upper())
        args.name_src = names[args.s][0].upper()
        if not osp.exists(args.output_dir_src):
            os.system('mkdir -p ' + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            os.mkdir(args.output_dir_src)

        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        mddn_F, mddn_C1, mddn_C2,mddn_E1,mddn_E2=train_source_step1(args)
        train_source_step2(args,mddn_F, mddn_C1, mddn_C2,mddn_E1,mddn_E2) #add evi
        args.out_file = open(osp.join(args.output_dir_src, 'log_test_transform.txt'), 'w')
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            folder ='/home/spi/peijiangbo/'
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            test_target(args)
