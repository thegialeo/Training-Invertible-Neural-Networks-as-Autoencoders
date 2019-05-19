import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
from functionalities import filemanager as fm
from functionalities import MMD_autoencoder_loss as cl


def get_loss(loader, model, criterion, latent_dim, tracker, conditional=False, disc_lst=None, use_label=False, device='cpu'):
    """
    Compute the loss of a model on a train, test or evalutation set wrapped by a loader.

    :param loader: loader that wraps the train, test or evaluation set
    :param model: model that should be tested
    :param criterion: the criterion to compute the loss
    :param latent_dim: dimension of the latent space
    :param tracker: tracker for values during training
    :param disc_lst: If given the first latent dimension will be enforced to be discrete depending on the values given
    in disc_lst
    :param use_label: If true, the labels will be used to help enforcing the first latent dimension to be discrete
    :param device: device on which to do the computation (CPU or CUDA). Please use get_device() function to get the
    device, if using multiple GPU's. Default: cpu
    :return: losses
    """

    model.to(device)

    model.eval()

    if disc_lst is not None or conditional:
        losses = np.zeros(6, dtype=np.double)
    else:
        losses = np.zeros(5, dtype=np.double)

    tracker.reset()
    
    correct = 0

    for i, data in enumerate(tqdm(loader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            lat_img = model(inputs)
            lat_shape = lat_img.shape
            lat_img = lat_img.view(lat_img.size(0), -1)

            if conditional:
                binary_label = lat_img.new_zeros(lat_img.size(0), 10)
                idx = torch.arange(labels.size(0), dtype=torch.long)
                binary_label[idx, labels] = 1
                lat_img_mod = torch.cat([lat_img[:,:latent_dim], binary_label, lat_img.new_zeros((lat_img[:,latent_dim+10:]).shape)], dim=1)
                pred = lat_img[:, latent_dim:latent_dim+10].max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
            elif use_label and disc_lst is not None:
                disc_lst = torch.tensor(disc_lst).to(device)
                disc_lat_dim = disc_lst[labels]
                lat_img_mod = torch.cat([torch.unsqueeze(disc_lat_dim, 1), lat_img[:, 1:latent_dim],
                                         lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)
                pred = disc_lst[torch.min(torch.abs(lat_img[:, :1] - disc_lst), 1)[1]] * 10
                correct += pred.eq(labels.float().view_as(pred)).sum().item()
            elif disc_lst is not None:
                disc_lst = torch.tensor(disc_lst).to(device)
                disc_lat_idx = torch.min(torch.abs(lat_img[:, :1] - disc_lst), 1)[1]
                disc_lat_dim = disc_lst[disc_lat_idx]
                lat_img_mod = torch.cat([torch.unsqueeze(disc_lat_dim, 1), lat_img[:, 1:latent_dim],
                                         lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)
            else:
                lat_img_mod = torch.cat([lat_img[:, :latent_dim], lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)

            lat_img_mod = lat_img_mod.view(lat_shape)

            output = model(lat_img_mod, rev=True)

            if use_label:
                batch_loss = criterion(inputs, lat_img.to(device), output.to(device), labels)
            else:
                batch_loss = criterion(inputs, lat_img.to(device), output.to(device))

            for i in range(len(batch_loss)):
                losses[i] += batch_loss[i].item()

            tracker.update(lat_img)
            
    correct = correct * 100. / len(loader.dataset)
    if use_label or conditional:
        print('Test Accuracy: {:.1f}'.format(correct))
    losses /= len(loader)
    return losses


def get_loss_bottleneck(loader, modelname, subdir, latent_dim_lst, device, a_distr, a_rec, a_spar, a_disen):
    """


    :return:
    """

    total_loss = []
    rec_loss = []
    dist_loss = []
    spar_loss = []
    disen_loss = []

    for i in latent_dim_lst:
        print('bottleneck dimension: {}'.format(i))
        model = fm.load_model('{}_{}'.format(modelname, i).to(device), subdir)
        criterion = cl.MMD_autoencoder_loss(a_distr=a_distr, a_rec=a_rec, a_spar=a_spar, a_disen=a_disen, latent_dim=i, loss_type='l1', device=device)
        losses = get_loss(loader, model, criterion, i, device)
        total_loss.append(losses[0])
        rec_loss.append(losses[1])
        dist_loss.append(losses[2])
        spar_loss.append(losses[3])
        disen_loss.append(losses[4])

    return total_loss, rec_loss, dist_loss, spar_loss, disen_loss