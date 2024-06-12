from torch.utils.data import Dataset
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
# from disentangle import resnet18
# from disentangle_test import resnet18
from disentangle_mhsa import resnet18
# from disentangle_cbam import resnet18
# from disentangle_sa import sa_resnet18
# from disentangle_eca import eca_resnet18
# from disentangle_test_b4_eca import resnet18
# from disentangle_test_b4 import resnet18
# from disentangle_b4_mhsa import resnet18
#from disentangle_b3_mhsa_ace import resnet18
# from disentangle_ca import ca_resnet18
# from bkb_model.disentangle_bkb_2 import resnet18
import torch.optim as optim
from until import train_one_epoch_two,evaluate_two
from datetime import datetime
import pandas as pd
import torchio as tio
from collections.abc import Iterable

start = datetime.now()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loading workers.')
    parser.add_argument('--tensor_root', default='', help='tensorboard save path')
    parser.add_argument('--data_path', type=str,default="")
    parser.add_argument('--val_data_path', type=str, default="")
    parser.add_argument('--train_data_path', type=str, default="")
    parser.add_argument('--weights1', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--weights2', type=str, default='')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=140,help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size.')
    parser.add_argument('--w1', type=int, default=1)
    parser.add_argument('--w2', type=int, default=0.1)
    parser.add_argument('--tag', type=str, default='real_ol')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of class.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for Adam.')
    parser.add_argument('--heads', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--double', type=bool, default=True)

    return parser.parse_args()

def freeze_layer(model,layer_names,freeze=False):
    if not isinstance(layer_names,Iterable):
        layer_names=[layer_names]
    for name,child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze




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

def get_classes(root:str):
    mri_class=[cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    mri_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(mri_class))
    return class_indices

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    folds = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

    for fold in folds:
        tb_writer = SummaryWriter(log_dir=args.tensor_root)
        data_aug = tio.Compose([#tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
                                #tio.RandomAffine(scales=0.1, degrees=0, translation=0),
                                tio.RandomAffine(scales=0, degrees=30, translation=5)
                                ])
        classes = get_classes(args.data_path)
        val_data_path = args.val_data_path  + f"/new_validation_split{fold}.csv"
        train_data_path = args.train_data_path  + f"/new_train_split{fold}.csv"
        train_datasets = Get_Dataset(train_data_path,data_aug,True)
        val_datasets = Get_Dataset(val_data_path)

        best_acc=0.0
        batch_size = args.batch_size
        num_workers = args.num_workers
        resolution = np.array([160.,160.,160.])

        train_dataloader = torch.utils.data.DataLoader(
            train_datasets,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_datasets,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False
        )

        # model = ca_resnet18(no_cuda=device).to(device)
        model = resnet18(num_classes=args.num_classes, no_cuda=device, resolution=resolution, heads=args.heads).to(
            device)
        # model = sa_resnet18(num_seg_classes=args.num_classes,no_cuda=args.device).to(device)
        # model = resnet18(num_classes=args.num_classes).to(device)

        model_dict = model.state_dict()
        weight_dict = {}
        if args.weights1 != "":
            assert os.path.exists(args.weights1), "weights file: '{}' not exist.".format(args.weights1)
            print('loading pretrained model {}'.format(args.weights1))
            pretrain = torch.load(args.weights1, map_location=device)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items()}

            weight_dict.update(pretrain_dict)

        if weight_dict:
            keys = []
            for k, v in weight_dict.items():
                keys.append(k)
            i = 0
            for k, v in model_dict.items():
                if i >= len(keys):
                    break
                if v.size() == weight_dict[keys[i]].size():
                    model_dict[k] = weight_dict[keys[i]]
                    i = i + 1

        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        print('missing_keys:', *[missing_keys], sep='\n')
        print('unexpected_keys:', *[unexpected_keys], sep='\n')

        learning_rate = args.lr
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=learning_rate, weight_decay=args.weight_decay)
        epochs = args.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
        for epoch in range(epochs):
            # train
            train_loss, train_acc, accuracy_dict, train_class_loss, train_regress_loss, train_ol_loss = train_one_epoch_two(
                                                                                                    model=model,
                                                                                                    optimizer=optimizer,
                                                                                                    data_loader=train_dataloader,
                                                                                                    device=device,
                                                                                                    epoch=epoch,
                                                                                                    classes=classes,
                                                                                                    w1=args.w1,
                                                                                                    w2=args.w2,
                                                                                                    tag=args.tag,
                                                                                                    double=args.double)

            scheduler.step()

            val_loss, val_acc, accuracy_dict, val_class_loss, val_regress_loss, MAE, r2_score, val_ol_loss, accuracy_dict_age = evaluate_two(
                                                                                                                    model=model,
                                                                                                                    data_loader=val_dataloader,
                                                                                                                    device=device,
                                                                                                                    epoch=epoch,
                                                                                                                    classes=classes,
                                                                                                                    w1=args.w1,
                                                                                                                    w2=args.w2,
                                                                                                                    tag=args.tag)


            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "ad", "nc", 'mci',]
            tags2=["train_class_loss","train_regress_loss","val_class_loss","val_regress_loss","MAE","r2_score","train_ol_loss","val_ol_loss"]
            tags3=["ad_age","nc_age","mci_age"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[5], accuracy_dict[tags[5]], epoch)
            tb_writer.add_scalar(tags[6], accuracy_dict[tags[6]], epoch)
            tb_writer.add_scalar(tags[7], accuracy_dict[tags[7]], epoch)

            tb_writer.add_scalar(tags2[0], train_class_loss, epoch)
            tb_writer.add_scalar(tags2[1], train_regress_loss, epoch)
            tb_writer.add_scalar(tags2[2], val_class_loss, epoch)
            tb_writer.add_scalar(tags2[3], val_regress_loss, epoch)
            tb_writer.add_scalar(tags2[4], MAE, epoch)
            tb_writer.add_scalar(tags2[5], r2_score, epoch)
            tb_writer.add_scalar(tags2[6],train_ol_loss, epoch)
            tb_writer.add_scalar(tags2[7],val_ol_loss, epoch)

            tb_writer.add_scalar(tags3[0], accuracy_dict_age[tags[5]], epoch)
            tb_writer.add_scalar(tags3[1], accuracy_dict_age[tags[6]], epoch)
            tb_writer.add_scalar(tags3[2], accuracy_dict_age[tags[7]], epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                print('best_acc:', best_acc)
                if val_acc>=0.68:
                    torch.save(model.state_dict(),
                               f"./weights/final_experiment/double/bkb_test/bkb_2/dis_nool-{fold}-{round(best_acc, 4)}-{epoch}.pth")
                    print('Best Model saved.')
        print('finished!')
    stop = datetime.now()
    print("Running time: ", stop - start)

if __name__ == '__main__':
    opt=parse_args()
    main(opt)





