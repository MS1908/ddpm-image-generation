from torchvision import datasets, transforms
from torch.utils import data


def dataset_factory(dataset_name, dataset_root, bs=32):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root=dataset_root, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4915, 0.4823, 0.4468),
                                 std=(0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(root=dataset_root, download=True, transform=transform)
    else:
        dataset = None

    n_classes = len(dataset.classes)
    dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)

    return dataloader, n_classes
