import os
import numpy as np
import pandas as pd
import torch

from my_utils_2 import GradCAM
from pathlib import Path

from disentangle_mhsa import resnet18
import cv2
import random

def my_show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      save_path=None,
                      colormap: int = cv2.COLORMAP_JET):
    img_ = img.squeeze(0)
    mask_ = mask.squeeze(0)

    for s in range(img_.shape[0]):
        mask = mask_[s, :, :]
        img = img_[s, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = heatmap.astype(float)
        heatmap = heatmap / 255
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
        cam = heatmap + img.astype(float)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        write_path = save_path +'/'+ 'x_' +  str(s) + '.png'
        png_compression = 3
        cv2.imwrite(write_path,cam,[cv2.IMWRITE_PNG_COMPRESSION,png_compression])

    for s in range(img_.shape[1]):
        mask = mask_[:, s, :]
        img = img_[:, s, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = heatmap.astype(float)
        heatmap = heatmap / 255
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
        cam = heatmap + img.astype(float)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        write_path = save_path +'/'+ 'y_' +  str(s) + '.png'
        cv2.imwrite(write_path, cam, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

    for s in range(img_.shape[2]):
        mask = mask_[:, :, s]
        img = img_[:, :, s]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = heatmap.astype(float)
        heatmap = heatmap / 255
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
        cam = heatmap + img.astype(float)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        write_path = save_path +'/'+ 'z_' +  str(s) + '.png'
        cv2.imwrite(write_path, cam, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

def get_path(data_path):
    for root,dir,file in os.walk(data_path):
        if 'ad' in os.path.basename(root):
            ad_images = [os.path.join(root,i) for i in os.listdir(root)]
            ad_path = random.sample(ad_images,k=5)
        if 'mci' in os.path.basename(root):
            mci_images = [os.path.join(root,i) for i in os.listdir(root)]
            mci_path = random.sample(mci_images,k=5)
        if 'nc' in os.path.basename(root):
            nc_images = [os.path.join(root,i) for i in os.listdir(root)]
            nc_path = random.sample(nc_images,k=5)
    return ad_path,mci_path,nc_path


def get_weights(path,include=None):
    weight_list = []
    for root,_,files in os.walk(path):
        for file in files:
            if include is None or Path(file).suffix == include:
                weight_list.append(root+"/"+file)
    weight_list.sort()
    return weight_list


def main():
    # model=resnet18(num_seg_classes=3).to('cuda:1')
    model =resnet18(num_classes=3, no_cuda="cuda:1", resolution=np.array([160.,160.,160.]), heads=4).to('cuda:1')
    weight_list = get_weights("./best_weight_pl",include='.pth')
    for fold_step,weight_path in enumerate(weight_list):
        print(f'this is {fold_step} fold')
        model.load_state_dict(torch.load(weight_path,map_location=torch.device('cuda:1')))
        print(f'weight_path {weight_path[-24:]}')

        # target_layers = [model.branch_2.layer4]
        # target_layers = [model.layer3,model.branch_2.layer4]
        # target_layerss = {'b4':[model.branch_2.layer4],
        #                  'b2':[model.layer2],
        #                   'b3': [model.layer3]}
        target_layerss = {'b3+b4':[model.layer3,model.layer4]}

        # ad_path,mci_path,nc_path = get_path('/media/gablab/home/syl/dataset')
        data_path = f"./data/val_data_5_{fold_step}.csv"
        # data_path = "./two.csv"
        df = pd.read_csv(data_path)
        print(data_path[-16:])
        ad_path = df[df['class'] == 0]['path'].tolist()
        print(len(ad_path))
        # ad_path = random.sample(ad_path, k=70)
        mci_path = df[df['class'] == 1]['path'].tolist()
        print(len(mci_path))
        # mci_path = random.sample(mci_path, k=40)
        nc_path = df[df['class'] == 2]['path'].tolist()
        print(len(nc_path))
        # nc_path = random.sample(nc_path, k=50)
        mri_path=[ad_path,mci_path,nc_path]
        # scales ={"0.7,0.3":np.array([0.7,0.3]),"0.3,0.7":np.array([0.3,0.7])}
        scales = {#"0.7,0.3": np.array([0.7, 0.3]),
        	   #"0.6,0.4": np.array([0.6, 0.4]),
        	   "0.5,0.5": np.array([0.5, 0.5]),
        	   #"0.4,0.6": np.array([0.4, 0.6]),
        	   #"0.3,0.7": np.array([0.3, 0.7]),
               "0.0,1.0": np.array([0.0, 1.0]),
               "1.0,0.0": np.array([1.0, 0.0])
                }
        for scale_step,scale_value in scales.items():
            print(f"scale_value {scale_value}")
            for target_name,target_layers in target_layerss.items():
                print(target_name)
                for step , name in enumerate(mri_path):
                    for step2 , path in enumerate(name):
                        img_path = path
                        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

                        img= np.load(img_path)
                        # print(img.shape)
                        input_tensor = torch.from_numpy(img)
                        input_tensor.unsqueeze_(dim=0)#.unsqueeze_(dim=0)
                        # print('input_size',input_tensor.shape)

                        cam_base = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
                        # cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
                        # cam = RandomCAM(model=model, target_layers=target_layers, use_cuda=True)
                        # cam_hires = HiResCAM(model=model, target_layers=target_layers, use_cuda=True)
                        # cam_grad = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
                        # cam_layer = LayerCAM(model=model, target_layers=target_layers, use_cuda=True)

                        target_category = step
                        # target_category = None

                        grayscale_cam = cam_base(input_tensor=input_tensor,scales=scale_value, target_category=target_category)
                        # grayscale_cam1 = cam_hires(input_tensor=input_tensor, target_category=target_category)
                        # grayscale_cam2 = cam_grad(input_tensor=input_tensor, target_category=target_category)
                        # grayscale_cam3 = cam_layer(input_tensor=input_tensor, target_category=target_category)

                        # grayscale_cam = grayscale_cam[0, :]
                        # print(grayscale_cam.shape)
                        print(step,':',step2)
                        #visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                        #                                   grayscale_cam,
                        #                                   use_rgb=True)
                        #print('max',np.max(img))
                        img=(img-np.min(img))/np.max(img)

                        save_path = f'./cam/fold_{fold_step}/{scale_step}/{target_name}/label{step}/{os.path.basename(path)}'
                        # save_path_hires = f'./cam/hires/{target_name}/{step}/{os.path.basename(path)}'
                        # save_path_grad = f'./cam/grad/{target_name}/{step}/{os.path.basename(path)}'
                        # save_path_layer = f'./cam/layer/{target_name}/{step}/{os.path.basename(path)}'
                        # if os.path.exists(save_path_hires) is False:
                        #     os.makedirs(save_path_hires)
                        # if os.path.exists(save_path_grad) is False:
                        #     os.makedirs(save_path_grad)
                        # if os.path.exists(save_path_layer) is False:
                        #     os.makedirs(save_path_layer)
                        if os.path.exists(save_path) is False:
                            os.makedirs(save_path)

                        my_show_cam_on_image(img,
                                              grayscale_cam,
                                             save_path
                                              )

                        # my_show_cam_on_image(img,
                        #                       grayscale_cam1,
                        #                      save_path_hires
                        #                       )#image_name='test'
                        # my_show_cam_on_image(img,
                        #                       grayscale_cam2,
                        #                      save_path_grad
                        #                       )
                        # my_show_cam_on_image(img,
                        #                       grayscale_cam3,
                        #                      save_path_layer
                        #                       )
                        # plt.imshow(visualization)
                        # plt.show()


if __name__ == '__main__':
    main()
