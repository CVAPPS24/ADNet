import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np


class OrthogonalLoss(torch.nn.Module):
    def __init__(self,tag):
        super(OrthogonalLoss, self).__init__()
        self.tag=tag
    def forward(self, tensor_1, tensor_2,labels,device):
        temp1,temp2=torch.zeros(1).to(device),torch.zeros(1).to(device)
        temp = torch.zeros(1).to(device)
        num=torch.zeros(1).to(device)
        if self.tag=='nc_ol':
            for step,idx in enumerate(labels):
                if idx.item() == 2:
                    temp1+=1-torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0)
                    num+=1
                else:
                    pass

            return (temp1+temp2)/(num+1e-7)
        elif self.tag=='ad_mci_ol':
            for step, idx in enumerate(labels):
                if idx.item() == 2:
                    pass
                else:
                    temp2+=torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))
                    num += 1

            return (temp1 + temp2) / (num+1e-7)
        elif self.tag=='real_ol_2':
            for step, idx in enumerate(labels):
                temp+=max(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0),0)
                return temp
        elif self.tag=='mci_ol':
            for step, idx in enumerate(labels):
                if idx.item() == 2 or idx.item() == 0:
                    temp1 += 1 - torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0)
                else:
                    temp2 += torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))

            return (temp1 + temp2) / tensor_1.shape[0]
        elif self.tag=='real_ol':
            for step, idx in enumerate(labels):
                temp+=torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))
                return temp
        else:
            for step, idx in enumerate(labels):
                if idx.item() == 2:
                    temp1 += 1 - torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0)
                else:
                    temp2+=torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))

            return (temp1 + temp2) / tensor_1.shape[0]

class Smooth_L1_Loss(torch.nn.Module):
    def __init__(self,beta=1.0,reduction='mean'):
        super(Smooth_L1_Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    def forward(self,input_tensor,target_tensor,label_tensor):
        abs_diff = torch.abs(input_tensor - target_tensor)
        condition_0 = (label_tensor == 0) & (abs_diff <= 5)
        condition_1 = (label_tensor == 1) & (abs_diff <= 3)
        condition_2 = label_tensor == 2

        modified_diff = torch.where(condition_0 | condition_1, torch.zeros_like(abs_diff), abs_diff)
        cond = modified_diff < self.beta
        loss = torch.where(cond, 0.5 * modified_diff ** 2 / self.beta, modified_diff - 0.5 * self.beta)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def metric(output, target):
    pred = output.cpu()
    target = target.cpu()
    mae = mean_absolute_error(target,pred)
    r2=r2_score(target,pred)
    return mae,r2


def train_one_epoch_two(model, optimizer, data_loader, device, epoch,classes,w1=0.5,w2=0.5,tag="all_ol",double=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function_r = Smooth_L1_Loss()
    loss_function_o = OrthogonalLoss(tag)
    accu_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)
    regress_loss = torch.zeros(1).to(device)
    ol_loss = torch.zeros(1).to(device)

    regress_loss_with_weight = torch.zeros(1).to(device)
    ol_loss_with_weight = torch.zeros(1).to(device)
    all_no_weight_loss_sum = torch.zeros(1).to(device)

    correct_pred = {classname: 0 for classname in classes}
    correct_pred_age={classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    class_key=dict((v,k) for v, k in enumerate(classes))

    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    w3 = 0.1
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        if double ==True:
            images,images_aug, labels, labels2 = data
            images = torch.cat([images, images_aug], 0)
            labels = torch.cat([labels, labels], 0)
            labels2 = torch.cat([labels2, labels2], 0)
        else:
            images, labels,labels2= data
        sample_num += images.shape[0]
        labels2=labels2.to(torch.float32)

        pred_r,pred_c,tensor_1,tensor_2 = model(images.to(device))
        pred_classes = torch.max(pred_c, dim=1)[1]

        for label, prediction,label2,tags in zip(labels, pred_classes,labels2,pred_r):
            label=label.to(device)
            label2 = label2.to(device)

            if label == prediction:
                correct_pred[class_key[label.item()]] += 1
            if label2 < (tags+5) and label2 > (tags-5):
                correct_pred_age[class_key[label.item()]] += 1
            total_pred[class_key[label.item()]] += 1


        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss1 = loss_function(pred_c, labels.to(device))
        loss2 = loss_function_r(pred_r, labels2.to(device),labels.to(device))

        loss3 = loss_function_o(tensor_1,tensor_2,labels.to(device),device)

        if tag=='no_ol':
            loss = w1*loss1+w2*loss2
        elif tag =='real_ol' and epoch>30:
            loss = w1*loss1+w2*loss2+w3*loss3
        else:
            loss = w1*loss1+w2*loss2
        loss.backward()


        accu_loss += loss.detach()
        class_loss += loss1.detach()
        regress_loss += loss2.detach()
        ol_loss += loss3.detach()

        regress_loss_with_weight += (loss2.detach()* w2)
        ol_loss_with_weight += (loss3.detach()* w3)
        all_no_weight_loss_sum +=(loss1.detach() + loss2.detach() + loss3.detach())



        data_loader.desc = "[train epoch {}] loss: {:.3f}, c_acc: {:.3f} ".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num,
                                                                                )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    print('total_pred_end', total_pred)
    print('correct_pred_end', correct_pred)
    print('correct_pred_age', correct_pred_age)
    accuracy_dict = {}
    accuracy_dict_age = {}
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] <=0:
            accuracy=0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
        accuracy_dict[classname] = accuracy

    for classname, correct_count in correct_pred_age.items():
        if total_pred[classname] <=0:
            accuracy=0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print("age_pred_acc for class {:5s} is: {:.1f} %".format(classname, accuracy))
        accuracy_dict_age[classname] = accuracy

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,\
           accuracy_dict,class_loss.item() / (step + 1),\
           regress_loss.item() / (step + 1), ol_loss.item()/(step+1),\
           regress_loss_with_weight.item() / (step + 1), ol_loss_with_weight.item() / (step + 1),\
           all_no_weight_loss_sum.item() / (step + 1)


@torch.no_grad()
def evaluate_two(model, data_loader, device, epoch, classes=None,w1=0.5,w2=0.5,tag="all_ol"):
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function_r = Smooth_L1_Loss()
    loss_function_o = OrthogonalLoss(tag)

    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)
    regress_loss = torch.zeros(1).to(device)
    ol_loss = torch.zeros(1).to(device)

    regress_loss_with_weight = torch.zeros(1).to(device)
    ol_loss_with_weight = torch.zeros(1).to(device)
    all_no_weight_loss_sum = torch.zeros(1).to(device)

    MAE = torch.zeros(1).to(device)
    R2 = torch.zeros(1).to(device)

    if classes != None:
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        correct_pred_age = {classname: 0 for classname in classes}
        class_key=dict((v,k) for v, k in enumerate(classes))
        confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int16)


    sample_num = 0
    w3 = 0.1
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images,labels,labels2 = data
        sample_num += images.shape[0]
        labels2=labels2.to(torch.float32)

        pred_r,pred_c,tensor_1,tensor_2 = model(images.to(device))
        pred_classes = torch.max(pred_c, dim=1)[1]

        if classes!=None:
            for label, prediction, label2, tags in zip(labels, pred_classes, labels2, pred_r):
                label = label.to(device)
                label2 = label2.to(device)
                # print(label2, tag)
                confusion_matrix[prediction.item()][label.item()] += 1
                if label == prediction:
                    correct_pred[class_key[label.item()]] += 1
                if label2 <= (tags + 3) and label2 >= (tags - 3): #top3
                    correct_pred_age[class_key[label.item()]] += 1
                total_pred[class_key[label.item()]] += 1


        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss1 = loss_function(pred_c, labels.to(device))
        loss2 = loss_function_r(pred_r, labels2.to(device), labels.to(device))

        loss3 = loss_function_o(tensor_1, tensor_2, labels.to(device),device)
        if tag == 'no_ol':
            loss = w1*loss1 + w2* loss2
        else:
            loss = w1 * loss1 + w2 * loss2 + w3*loss3

        mae,r2 = metric(pred_r.detach(), labels2.detach().cpu())

        accu_loss += loss.detach()
        class_loss += loss1.detach()
        regress_loss += loss2.detach()
        ol_loss += loss3.detach()

        regress_loss_with_weight += (loss2.detach()* w2)
        ol_loss_with_weight += (loss3.detach()* w3)
        all_no_weight_loss_sum +=(loss1.detach() + loss2.detach() + loss3.detach())

        MAE += mae
        R2+=r2

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, c_acc: {:.3f} ,MAE:{:.3f},r2_scroe:{:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num,
                                                                               MAE.item(),
                                                                                R2.item()/(step + 1),
                                                                                )
    if classes!=None:
        print('total_pred_end', total_pred)
        print('correct_pred_end', correct_pred)
        print('correct_pred_age', correct_pred_age)
        print('confusion_matrix: ')
        print(confusion_matrix)

        accuracy_dict = {}
        accuracy_dict_age = {}
        for classname, correct_count in correct_pred.items():
            if total_pred[classname] <= 0:
                accuracy = 0
            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            accuracy_dict[classname] = accuracy

        for classname, correct_count in correct_pred_age.items():
            if total_pred[classname] <= 0:
                accuracy = 0
            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
            print("age_pred_acc class {:5s} is: {:.3f} %".format(classname, accuracy))
            accuracy_dict_age[classname] = accuracy

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num, \
               accuracy_dict,class_loss.item() / (step + 1),regress_loss.item() / (step + 1),MAE,\
               R2.item()/(step+1),ol_loss.item() / (step + 1),accuracy_dict_age, \
               regress_loss_with_weight.item() / (step + 1), ol_loss_with_weight.item() / (step + 1), \
               all_no_weight_loss_sum.item() / (step + 1)
    return accu_loss.item() / (step + 1) ,accu_num.item() / sample_num

