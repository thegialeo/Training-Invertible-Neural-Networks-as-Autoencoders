import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist(aug=False):
    """
    Check if the MNIST dataset already exists in the directory "./datasets/mnist". If not, the MNIST dataset is
    downloaded. Returns trainset, testset and classes of MNIST. Applied transformations: RandomResizedCrop(),
    RandomVerticalFlip(), RandomHorizontalFlip(), RandomRotation(), ToTensor().

    :return: trainset, testset, classes of MNIST
    """

    save_path = "./datasets/mnist"

    transform_train = transforms.Compose([transforms.RandomRotation(20), transforms.ColorJitter(),  transforms.ToTensor()]) # 

    transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(root=save_path, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root=save_path, train=False, transform=transform, download=True)

    classes = [x for x in range(10)]

    return trainset, testset, classes


def load_cifar():
    """
    Check if the CIFAR10 dataset already exists in the directory "./datasets/cifar". If not, the CIFAR10 dataset is
    downloaded. Returns trainset, testset and classes of CIFAR10. Applied transformations: RandomResizedCrop(),
    RandomVerticalFlip(), RandomHorizontalFlip(), RandomRotation(), ToTensor().

    :return: trainset, testset, classes of CIFAR10
    """

    save_path = "./datasets/cifar"

    transform_train = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 2)),
                                          transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(root=save_path, train=True, transform=transform, download=True)
    testset = datasets.CIFAR10(root=save_path, train=False, transform=transform, download=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return trainset, testset, classes


def get_loader(dataset, batch_size, pin_memory=True):
    """
    Create loader for a given dataset.

    :param dataset: dataset for which a loader will be created
    :param batch_size: size of the batch the loader will load during training
    :param pin_memory: pin_memory argument for pytorch dataloader, will be simply forwarded
    :return: loader
    """

    loader = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size)

    return loader


def split_dataset(dataset, ratio, batch_size, pin_memory=True):
    """
    Split a dataset into two subset. e.g. trainset and validation-/testset

    :param dataset: dataset, which should be split
    :param ratio: the ratio the two splitted datasets should have to each other
    :param pin_memory: pin_memory argument for pytorch dataloader, will be simply forwarded
    :return: dataloader_1, dataloader_2
    """

    indices = torch.randperm(len(dataset))
    idx_1 = indices[:len(indices) - int(ratio * len(indices))]
    idx_2 = indices[len(indices) - int(ratio * len(indices)):]

    dataloader_1 = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_1),
                                               num_workers=8, drop_last=True)

    dataloader_2 = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_2),
                                               num_workers=8, drop_last=True)

    return dataloader_1, dataloader_2


def make_dataloaders(trainset, testset, batch_size, valid_size=0, subset=None, pin_memory=True):
    """
    Create loaders for the train-, validation- and testset.

    :param trainset: trainset for which a train loader and valication loader will be created
    :param testset: testset for which we want to create a test loader
    :param batch_size: size of the batch the loader will load during training
    :param valid_size: size of the dataset wrapped by the validation loader
    :param subset: number of images per category (maximum: 5000 -> corresponds to whole training set)
    :param pin_memory: pin_memory argument for pytorch dataloader, will be simply forwarded
    :return: trainloader, validloader, testloader
    """

    if subset is None:
        indices = torch.randperm(len(trainset))
        train_idx = indices[:len(indices) - valid_size]
        valid_idx = indices[len(indices) - valid_size:]

        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=pin_memory, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
                                                  drop_last=True)

        validloader = torch.utils.data.DataLoader(trainset, pin_memory=pin_memory, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_idx),
                                                  drop_last=True)

        testloader = torch.utils.data.DataLoader(testset, pin_memory=pin_memory, batch_size=batch_size,
                                                 drop_last=True)

        return trainloader, validloader, testloader

    else:
        subset_idx = []
        counter = list(0.0 for i in range(10))

        for index, label in enumerate(trainset.train_labels):
            if counter[label] < subset:
                subset_idx.append(index)
            counter[label] += 1

        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_idx))

        testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=batch_size)

        return  trainloader, testloader