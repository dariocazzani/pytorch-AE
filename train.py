import argparse, os
import numpy as np
import imageio
from scipy import ndimage

import torch
from torchvision.utils import save_image

from models.VAE import VAE
from utils import get_interpolations

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if __name__ == "__main__":
    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)

    vae = VAE(args)
    for epoch in range(1, args.epochs + 1):
        vae.train(epoch)
        vae.test(epoch)

    with torch.no_grad():
        images, _ = next(iter(vae.test_loader))
        images = images.to(vae.device)
        images_per_row = 20
        interpolations = get_interpolations(vae.model, vae.device, images, images_per_row)

        sample = torch.randn(64, args.embedding_size).to(vae.device)
        sample = vae.model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                '{}/sample_VAE.png'.format(args.results_path))
        save_image(interpolations.view(-1, 1, 28, 28),
                '{}/interpolations_VAE.png'.format(args.results_path),  nrow=images_per_row)
        interpolations = interpolations.cpu()
        interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
        interpolations = ndimage.zoom(interpolations, 5, order=1)
        interpolations *= 256
        imageio.mimsave('{}/animation_VAE.gif'.format(args.results_path), interpolations.astype(np.uint8))
