import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
import torch.nn.functional as F

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            if hasattr(target_layer, 'register_full_backward_hook'): #不同版本兼容
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output): #收集当前网络层结构的输出
        activation = output
        self.activations.append(activation.cpu().detach()) #detach切断梯度

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x): #正向传播过程
        self.gradients = []
        self.activations = []
        return self.model(x.to("cuda:1"))[1]

    def release(self):
        for handle in self.handles:
            handle.remove()

def multilevel_spline_interpolation_torch(input_tensor_,new_size):
    input_array = input_tensor_.numpy()
    zoom_factors = [n/o for n,o in zip(new_size, input_array.shape)]
    return zoom(input_array,zoom_factors,order=3)

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=True):
        self.model = model.eval() #调到验证模式
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.to("cuda:1")
        self.activations_and_grads = ActivationsAndGradients(  #实现捕获特征层和反向传播权重
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """


    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3, 4), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]] #获取自己输入batch图片的感兴趣的类别的输出值相加
        return loss

    @staticmethod
    def get_loss_r(output):
        loss = 0
        for i in range(len(output)):
            loss = loss + output[i] #获取自己输入batch图片的感兴趣的类别的输出值相加
        return loss


    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads) #求权重
        # weights = self.get_cam_weights(activations,grads)  # 求权重
        weighted_activations = weights * activations  #[:, :, None, None]
        cam = weighted_activations.sum(axis=1) #求完和之后[B,D,H,W]

        return cam


    @staticmethod
    def get_target_width_height_depth(input_tensor):
        width,height,depth = input_tensor.size(-1),input_tensor.size(-2),input_tensor.size(-3)
        return width,height,depth

    def compute_cam_per_layer(self, input_tensor): #计算cam
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height_depth(input_tensor) #得到输入图片的深度宽度高度

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list): #遍历曾输出以及其对应的梯度信息
            cam = self.get_cam_image(layer_activations, layer_grads) #计算grad-cam
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)#this step is like no this step
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer,scales):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # result = np.mean(cam_per_target_layer, axis=1)
        # scales = np.array([0.7,0.3])
        scales = scales.reshape(1,-1,1,1,1)
        # scales = torch.tensor([0.7,0.3])
        # result = torch.einsum('ncdhw,c->ndhw',cam_per_target_layer,scales).unsqueeze(0)
        result = np.sum(cam_per_target_layer * scales,axis=1)
        return self.scale_cam_image(result)

    def __resize_data__(self, data, reshape_size):
        [depth, height, width] = data.shape
        scale = [reshape_size[0] * 1.0 / depth, reshape_size[1] * 1.0 / height, reshape_size[2] * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)  # 样条插值

        return data

    @staticmethod
    def scale_cam_image(cam, target_size=None): #后处理专用
        result = []
        for img in cam:
             #缩放到0，1之间，对cam做的
            if target_size is not None:
                img=torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                # img_ = multilevel_spline_interpolation_torch(img,target_size)
                img_ = F.interpolate(img, size=target_size, mode='trilinear',align_corners=False).squeeze(0).squeeze(0)
                # output_slices =[F.interpolate(slice_, size=target_size,mode='bilinear',align_corners=False)
                #                for slice_ in img.permute(2,0,1,3,4)]
                # img_ = torch.stack(output_slices,dim=2)
                img = img_.numpy()
                img = img - np.min(img)
                img = img / (1e-7 + np.max(img)) #img is in (0,1)
            result.append(img)
        result = np.float32(result)
        return result

    def __call__(self, input_tensor,scales, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.to("cuda:1")

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0) #one pic

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        # loss = self.get_loss_r(output)
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer,scales)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img
