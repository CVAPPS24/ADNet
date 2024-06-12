# coding:utf-8
import matplotlib.pyplot as plt
import torch
from disentangle_model.disentangle_mhsa import resnet18
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import sys
import os
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE



class Get_Dataset(Dataset):

    def __init__(self, data_path,transforms=None,double=None):
        self.subjects = pd.read_csv(data_path)
        self.index = list(self.subjects.columns.values)
        self.index.pop(0)
        self.transforms = transforms
        self.double = double

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
            tensor_data_aug = self.transforms(tensor_data)
            if self.double is not None:
                return tensor_data, tensor_data_aug, label1, label2
            else:
                return tensor_data_aug, label1, label2
        else:
            return tensor_data,label1,label2

def get_class_name(root:str):
    mri_class=[cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    mri_class.sort()
    class_indices =[ v for v, k in enumerate(mri_class)]
    class_indices_dict = dict((v, k) for v, k in enumerate(mri_class))
    return class_indices,class_indices_dict

features = []
def hook_fn(module,input,output):
    features.append(output.clone().detach())

def mean_all(features):
    for step,i in enumerate(features):
        temp = torch.mean(i.view(i.size(0),i.size(1),-1),dim=2)
        if step==0:
            result=temp.detach()
        else:
            result = torch.cat((result,temp),dim=0)
    return result.cpu().numpy()

def get_weights(path):
    weight_list = []
    for root,_,files in os.walk(path):
        for file in files:
            weight_list.append(root+"/"+file)
    weight_list.sort()
    return weight_list




def main():
    global features
    pre_save = False
    dir_ = 'mhsa_real_ol_double_rp'
    k_ = 'rp5'

    device ='cuda:1' if torch.cuda.is_available() else 'cpu'
    resolution = np.array([160.,160.,160.])
    heads = 5
    fold_id = 5
    if k_!='':
        weights = f"./checkpoint/{dir_}/{k_}"
    else:
        weights = f"./weights/pl_experiment/{dir_}/{k_}"
    weights = get_weights(weights)
    val_path = " "
    val_data_path = get_weights(val_path)
    val_data_path = val_data_path
    data_path=" "
    classes, classes_dict = get_class_name(data_path)
    classes = np.unique(classes)
    print('classes: ', classes)
    softmax = torch.nn.Softmax(dim=1)
    folds = [0]

    if pre_save == False:
        for fold in folds:
            pre_feature = torch.Tensor([0]).to(device)
            avg_feature = torch.Tensor([0]).to(device)
            model = resnet18().to(device)
            target_layer = model.maxpool
            hook = target_layer.register_forward_hook(hook_fn)
            print(weights[fold])
            print(val_data_path[fold])
            missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights[fold], map_location=device),
                                                                  strict=False)
            print('missing_keys:', *[missing_keys], sep='\n')
            print('unexpected_keys:', *[unexpected_keys], sep='\n')



            val_datasets = Get_Dataset(val_data_path[fold])

            val_dataloader = torch.utils.data.DataLoader(
                val_datasets,
                batch_size=8,
                pin_memory=False,
                num_workers=4,
                shuffle=False,
                drop_last=False)
            model.eval()
            with torch.no_grad():
                val_dataloader = tqdm(val_dataloader,file = sys.stdout)
                for step,data in enumerate(val_dataloader):
                    images,label1,_ = data
                    _,_,_,tensor = model(images.to(device))
                    # tensor = softmax(tensor)
                    if step ==0 :
                        avg_feature = tensor.detach()
                    else:
                        avg_feature =torch.cat((avg_feature,tensor.detach()),dim=0)
                    # avg_feature.append(np.array(tensor.clone().detach().cpu()).squeeze())
                pre_feature = mean_all(features)
            hook.remove()
            print('features.shape: ', pre_feature.shape)
            avg_feature = np.array(avg_feature.cpu())
            # if not os.path.isdir(f'./output/{dir_}/{k_}'):
            #     os.makedirs(f'./output/{dir_}/{k_}')
            # np.savetxt(f"./output/{dir_}/{k_}/{fold}.txt",avg_feature)
            print('avg_feature.shape: ',avg_feature.shape)
            if not os.path.isdir(f'./sne_save/{dir_}/{k_}'):
                os.makedirs(f'./sne_save/{dir_}/{k_}')
            np.save(f'./sne_save/{dir_}/{k_}/tl_intra_data_{fold}.npy',avg_feature)
            # np.save(f'./sne_save/{dir_}/{k_}/tl_intra_pre_data_{fold}.npy',pre_feature)
        # sys.exit()
    for k in folds:
    # k = 1
        if k_!='':
            avg_feature = np.load(f'./sne_save/{dir_}/{k_}/tl_intra_data_{k}.npy',allow_pickle=True)
        else:
            avg_feature = np.load(f'./sne_save/{dir_}/tl_intra_data_{k}.npy', allow_pickle=True)
        # pre_feature = np.load(f'./sne_save/{dir_}/{k_}/tl_intra_pre_data_{k}.npy',allow_pickle=True)


    #draw t-sne

        mark_list =['.','.','.']
        n_class = len(classes)
        palette = sns.hls_palette(n_class)
        tsne = TSNE(n_components=2,perplexity=30,n_iter=1000)
        x_tsne_2d = tsne.fit_transform(avg_feature)
        # x_tsne_2d = tsne.fit_transform(pre_feature)
        print("x_tsne_2d shape: ",x_tsne_2d.shape)

        df = pd.read_csv(val_data_path[k])
        print(val_data_path[k])
        edge = np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]])
        plt.figure(figsize=(13,13))
        # for coords in x_tsne_2d:
        #     plt.scatter(coords[0],coords[1],s=20)
        for idx,value in enumerate(classes):
            color = palette[idx]
            mark = mark_list[idx]
            indices = np.where(df['class']==value)
            print(indices)
            plt.scatter(x_tsne_2d[indices,0],x_tsne_2d[indices,1],color=color,marker=mark,label=classes_dict[value],s=250)
        plt.scatter(edge[:, 0], edge[:, 1], color='w')
        plt.legend(fontsize=16,markerscale=3,bbox_to_anchor=(1,1))
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.savefig(f'./sne_save/{dir_}/{k_}/t-sne-{k}.png',dpi=999)
        plt.show()

if __name__ == '__main__':
    main()













