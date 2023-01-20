import matplotlib.pyplot as plt
from PIL import Image
import torch


def get_fashion_mnist_labels(labels):
    """
    该函数将对应的标签转化为真实物品名称
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in labels]


def transform_convert(img_tensor, transform):
    """
    该函数将加载的图像数据转为普通图像，即逆操作
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    # 判断图像是否经过标准化操作，若是，则进行逆标准化操作还原图像
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    # 将tensor类型的图像数据的进行形状上的转换(通道x行x列---》行x列x通道)
    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    # 当图像执行了归一化操作时，对图像像素进行还原
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    # 当图像数据结构类型是tensor时，将其转为numpy类型
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    # 判断图像通道数目，并转换成对应类型图像
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        # 单通道输入二维数据，删除通道项，否则会报错
        img = Image.fromarray(img_tensor[:, :, 0].astype('uint8'))
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

def show_fashion_mnist(images, labels, transform):
    # labels = get_fashion_mnist_labels(labels)
    _,figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbs in zip(figs, images, labels):
        img = transform_convert(img, transform)
        f.imshow(img)
        f.set_title(lbs)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def SGD(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size