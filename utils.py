import numpy as np
import torch

def get_interpolations(args, model, device, images, images_per_row=20):
    model.eval()
    with torch.no_grad():
        def interpolate(t1, t2, num_interps):
            alpha = np.linspace(0, 1, num_interps+2)
            interps = []
            for a in alpha:
                interps.append(a*t2.view(1, -1) + (1 - a)*t1.view(1, -1))
            return torch.cat(interps, 0)

        if args.model == 'VAE':
            mu, logvar = model.encode(images.view(-1, 784))
            embeddings = model.reparameterize(mu, logvar).cpu()
        elif args.model == 'AE':
            embeddings = model.encode(images.view(-1, 784))
            
        interps = []
        for i in range(0, images_per_row+1, 1):
            interp = interpolate(embeddings[i], embeddings[i+1], images_per_row-4)
            interp = interp.to(device)
            interp_dec = model.decode(interp)
            line = torch.cat((images[i].view(-1, 784), interp_dec, images[i+1].view(-1, 784)))
            interps.append(line)
        # Complete the loop and append the first image again
        interp = interpolate(embeddings[i+1], embeddings[0], images_per_row-4)
        interp = interp.to(device)
        interp_dec = model.decode(interp)
        line = torch.cat((images[i+1].view(-1, 784), interp_dec, images[0].view(-1, 784)))
        interps.append(line)

        interps = torch.cat(interps, 0).to(device)
    return interps
