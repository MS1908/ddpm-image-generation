import argparse
import os
import torch
from torch import optim
from torchvision import utils
from tqdm import tqdm

from dataset_factory import dataset_factory
from ddpm import DDPM, ContextUnet


def train(
    dataloader,
    model,
    optimizer,
    learning_rate,
    epochs,
    device,
    model_dir=None,
    viz_dir=None,
    n_classes=10,
):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()

        optimizer.param_groups[0]['lr'] = learning_rate * (1 - epoch / epochs)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        avg_loss = 0.
        input_size = None
        for i, (images, labels) in pbar:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            input_size = (images.size(1), images.size(2), images.size(3))

            loss = model(images, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss /= len(dataloader)

        if model_dir:
            if best_loss > avg_loss:
                best_loss = avg_loss
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth'))

        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
            model.eval()
            with torch.no_grad():
                n_sample = 4 * n_classes

                x_gen = model.sample(n_sample, input_size, device, guide_w=2.0)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample / n_classes)):
                        try:
                            idx = torch.squeeze((labels == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * n_classes)] = images[idx]

                x_all = torch.cat([x_gen, x_real])
                # grid = utils.make_grid(x_all * -1 + 1, nrow=10)
                grid = utils.make_grid(x_all, nrow=10)
                utils.save_image(grid, os.path.join(viz_dir, f"image_ep{epoch}.png"))

    if model_dir:
        torch.save(model.state_dict(), os.path.join(model_dir, 'last.pth'))


N_FEATURES = 128
N_T = 400
LR = 1e-4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--epoch', type=int, help='Number of training epochs', default=10)
    parser.add_argument('--bs', type=int, help='Train batch size', default=128)
    parser.add_argument('--wt-path', type=str, help='Path to save the trained parameters of model',
                        default=None)
    parser.add_argument('--plot-path', type=str, help='Path to save the generate images in training',
                        default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, n_classes = dataset_factory(args.dataset, dataset_root='./data', bs=args.bs)

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=1, n_features=N_FEATURES, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=N_T,
        drop_prob=0.1,
        device=device
    )
    ddpm.to(device)

    optimizer = optim.Adam(ddpm.parameters(), lr=LR)

    train(
        dataloader=dataloader,
        model=ddpm,
        optimizer=optimizer,
        learning_rate=LR,
        epochs=args.epoch,
        device=device,
        model_dir=args.wt_path,
        viz_dir=args.plot_path,
        n_classes=n_classes
    )
