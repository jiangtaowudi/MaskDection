import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import Residual, ResNet18
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image



def main():
    # model = models.resnet50(pretrained=True)
    # target_layers = [model.layer4[-1]]

    model = ResNet18(Residual)
    target_layers = [model.b5[-1]]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])
    ])

    # Prepare image
    img_path = "u=2537911176,2928325435&fm=253&gp=0.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    # targets = [ClassifierOutputTarget(281)]     # cat
    targets = [ClassifierOutputTarget(0)]  # dog

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()