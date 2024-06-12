import sys

from torch.utils.data import Dataset
import argparse
import torch
import os
import numpy as np


# from disentangle import resnet18 as dual_net
from disentangle_mhsa import resnet18 as dual_net_mhsa
# from resnet import resnet18
# from disentangle_cbam import resnet18 as dual_net_cbam
# from disentangle_eca import eca_resnet18 as dual_net_eca
# from disentangle_sa import sa_resnet18 as dual_net_sa
# from disentangle_van import van_b0 as dual_net_van
# from disentangle_ca import ca_resnet18
# from bkb_model.disentangle_bkb_2 import resnet18 as bkb2
# from bkb_model.disentangle_bkb_4 import resnet18 as bkb4
# from disentangle_experiment_code.disentangle_model.unet import UNet3D
import torch.optim as optim
from until import evaluate_test
from datetime import datetime
import pandas as pd
import torchio as tio
# from AutomaticWeightedLoss import AutomaticWeightedLoss
# from mine import MINE


def parse_args():
    parser = argparse.ArgumentParser() #没有额外约束的单纯多任务,权重也没有调节过
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loading workers.')
    parser.add_argument('--data_path', type=str,default="")
    parser.add_argument('--val_data_path', type=str, default="")
    parser.add_argument('--weights1', type=str, default='./checkpoint',
                        help='initial weights path')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=1,help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size.')
    parser.add_argument('--tag', type=str, default='no_ol')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of class.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for Adam.')
    parser.add_argument('--heads', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--method', type=str, default='mhsa_rp')

    return parser.parse_args()


class Get_Dataset(Dataset):

    def __init__(self, data_path,transforms=None):
        self.subjects = pd.read_csv(data_path)
        self.index = list(self.subjects.columns.values)
        self.index.pop(0)
        self.transforms = transforms


    def __len__(self):
        return len(self.subjects[self.index[0]])

    def __getitem__(self, idx):
        img_path=self.subjects[self.index[0]][idx]
        img_path1 = img_path[:-18]
        img_path2 = img_path[-11:]
        img_path = img_path1 + img_path2
        img = np.load(img_path)
        assert img is not None
        group= self.subjects[self.index[1]][idx]
        age=self.subjects[self.index[2]][idx]

        img = img.astype(np.float32)
        tensor_data = torch.from_numpy(img)
        label1=group
        label2=age
        if self.transforms is not None:
            tensor_data = self.transforms(tensor_data)

        return tensor_data,label1,label2

def get_classes(root:str):
    mri_class=[cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    mri_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(mri_class))
    return class_indices

def get_temp(rp:int):
    if rp == 1:
        temp = 0
    if rp == 2:
        temp = 5
    if rp == 3:
        temp = 10
    return temp

def get_row_all(fold_score_all,mean_score_all):
    if fold_score_all.size == 0:
        fold_score_all = mean_score_all
    else:
        fold_score_all = np.row_stack((fold_score_all, mean_score_all))
    return fold_score_all

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    args.weights1= args.weights1 + f'/{args.method}'
    folds = [0,1,2,3,4]
    rps = [1,2]
    paths = []
    dirs_paths = []
    matrix_groups = []
    for root,dirs,files in os.walk(args.weights1):
        for name in files:
            paths.append(os.path.join(root,name))
        for name in dirs:
            dirs_paths.append(name)
    print('paths',paths)
    print('dirs_paths',dirs_paths)

    rp_score_all = np.array([])
    rp_auc_all = np.array([])
    rp_acc_all = np.array([])
    for rp in rps:
        temp = get_temp(rp)
        fold_score_all = np.array([])
        fold_auc_all = np.array([])
        fold_acc_all = np.array([])
        for fold in folds:
            classes = get_classes(args.data_path)
            val_data_path = args.val_data_path  + f"/new_validation_split{fold}.csv"
            val_datasets = Get_Dataset(val_data_path)

            num_workers = args.num_workers
            resolution = np.array([160.,160.,160.])

            val_dataloader = torch.utils.data.DataLoader(
                val_datasets,
                batch_size=args.batch_size,
                pin_memory=False,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False
            )
            if 'mhsa' in args.method:
                model = dual_net_mhsa(num_classes=args.num_classes,resolution=resolution,heads=args.heads).to(device)
            elif 'dual_net' in args.method:
                model = dual_net(num_classes=args.num_classes).to(device)
            elif 'resnet' in args.method:
                model = resnet18().to(device)
            elif 'cbam' in args.method:
                model = dual_net_cbam(num_seg_classes=args.num_classes,no_cuda=args.device).to(device)
            elif 'eca' in args.method:
                model = dual_net_eca(num_seg_classes=args.num_classes,no_cuda=args.device).to(device)
            elif 'sa' in args.method:
                model = dual_net_sa(num_seg_classes=args.num_classes,no_cuda=args.device).to(device)
            elif 'van' in args.method:
                model = dual_net_van(pretrained=False).to(device)
            elif 'ca' in args.method:
                model = ca_resnet18(no_cuda=args.device).to(device)
            elif 'bkb_2' in args.method:
                model = bkb2(num_classes=args.num_classes, no_cuda=args.device, resolution=resolution,
                                 heads=args.heads).to(device)
            elif 'bkb_4' in args.method:
                model = bkb4(num_classes=args.num_classes, no_cuda=args.device, resolution=resolution,
                                 heads=args.heads).to(device)
            if args.weights1 != "":
                missing_keys, unexpected_keys = model.load_state_dict(torch.load(paths[(fold+temp)],map_location=device),strict=False)
                print('missing_keys:', *[missing_keys], sep='\n')
                print('unexpected_keys:', *[unexpected_keys], sep='\n')

            val_acc,score_all,auc_,confusion_matrix = evaluate_test(  model=model,
                                                        data_loader=val_dataloader,
                                                        device=device,
                                                        epoch=0,
                                                        classes=classes,
                                                        fold=fold,
                                                        name='mhsa'
                                                        )

            df = pd.DataFrame(data=confusion_matrix,index=['ad','mci','nc'],columns=['ad','mci','nc'])
            df.loc[f'{fold}'] = np.NAN
            matrix_groups.insert(-1,df)
            mean_score_all = np.mean(score_all, axis=0)

            mean_auc = np.mean(auc_)
            if fold_score_all.size == 0:
                fold_score_all = mean_score_all
            else:
                fold_score_all = np.row_stack((fold_score_all,mean_score_all))
            fold_auc_all = np.append(fold_auc_all, mean_auc)
            fold_acc_all = np.append(fold_acc_all, val_acc)
        fold_score_all_mean = np.mean(fold_score_all, axis=0)
        fold_auc_all_mean = np.mean(fold_auc_all)
        fold_acc_all_mean = np.mean(fold_acc_all)
        fold_acc_all = np.append(fold_acc_all, fold_acc_all_mean)
        fold_auc_all = np.append(fold_auc_all, fold_auc_all_mean)

        rp_acc_all=get_row_all(rp_acc_all,fold_acc_all)
        print('rp_acc_all', rp_acc_all)
        rp_auc_all=get_row_all(rp_auc_all,fold_auc_all)
        print('rp_auc_all', rp_auc_all)
        rp_score_all=get_row_all(rp_score_all,fold_score_all_mean)
        print('rp_score_all',rp_score_all)

        fold_score_all = np.row_stack((fold_score_all,fold_score_all_mean))
        if (fold_score_all.shape[0],fold_score_all.shape[1]) == (6,4):
            final_result = pd.DataFrame(fold_score_all,index=[1,2,3,4,5,'mean'],columns=['SPE', 'SEN', 'PRE', 'F1'])
            final_result['AUC']=fold_auc_all
            print('final_result',final_result)
            if os.path.exists(f"./{args.method}/{rp}") is False:
                os.makedirs(f"./{args.method}/{rp}")
            final_result.to_csv(f'./{args.method}/{rp}/evaluate_result.csv')
            df2 = pd.concat(matrix_groups, ignore_index=False)
            df2.to_csv(f'./{args.method}/{rp}/matrix_result.csv')
        else:
            print('Error!')
            sys.exit(0)
    rp_score_all_mean = np.mean(rp_score_all, axis=0)
    rp_score_all = np.row_stack((rp_score_all, rp_score_all_mean))
    rp_auc_all_mean = np.mean(rp_auc_all, axis=0)
    rp_auc_all = np.row_stack((rp_auc_all, rp_auc_all_mean))
    rp_acc_all_mean = np.mean(rp_acc_all, axis=0)
    rp_acc_all = np.row_stack((rp_acc_all, rp_acc_all_mean))
    final_result_rp = pd.DataFrame(rp_score_all, index=['rp1', 'rp2', 'mean'], columns=['SPE', 'SEN', 'PRE', 'F1'])
    print('final_result_rp', final_result_rp)
    final_result_rp_auc = pd.DataFrame(rp_auc_all, index=['rp1', 'rp2', 'mean'], columns=[1,2,3,4,5,'mean'])
    print('final_result_rp_auc', final_result_rp_auc)
    final_result_rp_acc = pd.DataFrame(rp_acc_all, index=['rp1', 'rp2','mean'],columns=[1,2,3,4,5,'mean'])
    print('final_result_rp_acc', final_result_rp_acc)
    final_result_rp['AUC'] = rp_auc_all[:,5]
    final_result_rp['ACC'] = rp_acc_all[:,5]

    final_result_rp.to_csv(f'./{args.method}/evaluate_result.csv')
    final_result_rp_auc.to_csv(f'./{args.method}/evaluate_result_auc.csv')
    final_result_rp_acc.to_csv(f'./{args.method}/evaluate_result_acc.csv')

if __name__ == '__main__':
    opt=parse_args()
    main(opt)





