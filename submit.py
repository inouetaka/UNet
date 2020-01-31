import os

import torch
from PIL import Image
import numpy as np

from predict import predict_img
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def submit(net, device):
    dir = 'data/test/'

    N = len(list(os.listdir(dir)))
    with open('SUBMISSON.csv', 'a') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, device)
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i, ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).to(device)
    net.load_state_dict(torch.load('MODEL.pth', map_location=device))
    submit(net, device)