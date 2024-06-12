import os
import sys
import json
import pickle
import random
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

def curve_roc(pred, labels, save_name=''):
    plt.figure(figsize=(5, 5))
    colors = ['r', 'g', 'b']
    print('roc working...')
    roc_auc_3 = []
    for i in range(3):
        # for class i
        y_true = np.zeros(labels.shape)
        y_true[labels != i] = 0
        y_true[labels == i] = 1
        # label=1 -> positive   label=0 -> negative
        fpr, tpr, th = roc_curve(y_true=y_true, y_score=pred[:, i]) #use roc_curve method ?????
        roc_auc = auc(fpr, tpr)
        roc_auc_3.append(roc_auc) #1 label 1 roc 1 auc
        # print(roc_auc)
        plt.plot(fpr, tpr, color=colors[i])
    # plt.show()
    plt.savefig(save_name)
    return roc_auc_3

def evaluate_1(confusion_matrix, i):
    TP = confusion_matrix[i, i]
    FN = np.sum(confusion_matrix[i]) - TP
    FP = np.sum(confusion_matrix[:, i]) - TP
    TN = np.sum(confusion_matrix) - TP - FP - FN
    # print(TP, FP)
    # print(FN, TN)
    SPE = TN / (TN + FP)
    SEN = TP / (TP + FN)
    PRE = TP / (TP + FP)
    F1 = 2 * PRE * SEN / (PRE + SEN)
    return SPE, SEN, PRE, F1

def evaluate_mc(confusion_matrix,pred, labels,fold):
    score_all = []
    for idx in range(3):
        SPE, SEN, PRE, F1 =evaluate_1(confusion_matrix,idx)
        score_all.append([SPE, SEN, PRE, F1])
        print("evaluate_all_matrix:  ", np.mean(np.array(score_all), axis=0))
    auc_ = curve_roc(pred, labels, f'./picture/picture_{fold}')
    return score_all,auc_


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
                    # temp2+=torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))

            # print(temp1+temp2)
            return (temp1+temp2)/(num+1e-7) #tensor_1.shape[0]
            # return torch.sum(torch.abs(torch.cosine_similarity(tensor_1,tensor_2,dim=1)))/tensor_1.shape[0]
        elif self.tag=='ad_mci_ol':
            for step, idx in enumerate(labels):
                if idx.item() == 2:
                    # temp1 += 1 - torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0)
                    # num += 1
                    pass
                else:
                    temp2+=torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))
                    num += 1

            # print(temp1+temp2)
            return (temp1 + temp2) / (num+1e-7)  # tensor_1.shape[0]
        elif self.tag=='never_ol':
            for step, idx in enumerate(labels):
                temp1 += 1 - torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0)

            return (temp1 + temp2) / tensor_1.shape[0]  # tensor_1.shape[0]
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
            # print('error!!')
            # pass
            # return None
            for step, idx in enumerate(labels):
                if idx.item() == 2:
                    temp1 += 1 - torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0)
                else:
                    temp2+=torch.abs(torch.cosine_similarity(tensor_1[idx], tensor_2[idx], dim=0))

            return (temp1 + temp2) / tensor_1.shape[0]

@torch.no_grad()
def evaluate_test(model, data_loader, device, epoch, classes=None,fold=None,name=None,aibl=False):

    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    if classes != None:
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        correct_pred_age = {classname: 0 for classname in classes}
        class_key=dict((v,k) for v, k in enumerate(classes))
        confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int16)


    sample_num = 0
    # score_csv = open("score_csv.csv", 'w')
    predict_list=np.array([])
    label_list=np.array([])
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        if aibl == False:
            images, labels, labels2 = data
            labels2 = labels2.to(torch.float32)
        else:
            images, labels = data
        sample_num += images.shape[0]
        if 'resnet' in name:
             pred_c = model(images.to(device))
        else:
            pred_r,pred_c,tensor_1,tensor_2 = model(images.to(device))
        if predict_list.size == 0:
            predict_list = np.array(pred_c.cpu())
        else:
            predict_list = np.concatenate((predict_list,np.array(pred_c.cpu())),axis=0)
        label_list = np.append(label_list,labels.cpu()) #int(labels.cpu())
        pred_classes = torch.max(pred_c, dim=1)[1]

        if classes!=None:
            for label, prediction in zip(labels, pred_classes):
                # print('label:',label.item())
                # print('prediction',prediction.item())
                label = label.to(device)
                # label2 = label2.to(device)
                # print(label2, tag)
                confusion_matrix[prediction.item()][label.item()] += 1
                if label == prediction:
                    correct_pred[class_key[label.item()]] += 1
                # if label2 < (tags + 1) and label2 > (tags - 1):
                #     correct_pred_age[class_key[label.item()]] += 1
                total_pred[class_key[label.item()]] += 1


        accu_num += torch.eq(pred_classes, labels.to(device)).sum()


        data_loader.desc = "[valid epoch {}] c_acc: {:.3f} ".format(epoch,accu_num.item() / sample_num)
    if classes!=None:
        score_all, auc_ = evaluate_mc(confusion_matrix,predict_list, label_list,fold)
        score_all = np.array(score_all)
        auc_ = np.array(auc_)
    #just test result------
        print('total_pred_end', total_pred)
        print('correct_pred_end', correct_pred)
        print('pl',predict_list.shape)
        print('ll',label_list.shape)
        print('confusion_matrix: ')
        print(confusion_matrix)
        print('score_all: ',np.array(score_all))
        print('auc',auc_)
    #---------------
        return accu_num.item() / sample_num,score_all,auc_,confusion_matrix

