import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from functionalities import traverser as tra
from functionalities import dataloader as dl
from functionalities import filemanager as fm


def plot(x, y, x_label, y_label, plot_label, title, filename, sub_dim=None, figsize=(15, 10), font_size=24, y_log_scale=False):
    """
    Generate a plot based on given arguments. If y is a 2d numpy array/list, multiple plots will be generated within one
    diagram. If additionally x is also a 2d numpy array/list, multiple subplots will be generated.

    :param x: numpy array/list of x values to plot. If multiple subplots should be generated then x will be a 2d numpy
    array/list.
    :param y: numpy array/list of corresponding y values to plot. If multiple plots should be generated then y will be a
    2d numpy array/list.
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    :param plot_label: label for plots (appears in legend)
    :param title: title for the plot. Should be a list if multiple plots are generated.
    :param filename: file name under which the plot will be saved.
    :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
    :param figsize: the size of the generated plot
    :param font_size: font size of labels
    :return: None
    """

    plt.rcParams.update({'font.size': font_size})

    if not ('numpy' in str(type(x))):
        try:
            x = np.array(x)
        except TypeError:
            print("x is neither a numpy array nor a python list")

    if not ('numpy' in str(type(y))):
        try:
            y = np.array(y)
        except TypeError:
            print("y is neither a numpy array nor a python list")

    dim_x = len(x.shape)
    dim_y = len(y.shape)

    if (dim_x != 1 and dim_x != 2) or (dim_y != 1 and dim_y != 2) or (dim_x == 2 and dim_y == 1):
        raise ValueError("x has dimension {} and y has dimension {}".format(dim_x, dim_y))

    if dim_x == 1 and dim_y == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(x, y, label=plot_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_log_scale == True:
            ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True)
    elif dim_x == 1 and dim_y == 2:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for i, y_part in enumerate(y):
            ax.plot(x, y_part, label=plot_label[i])

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_log_scale == True:
            ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    elif dim_x == 2 and dim_y == 2:
        if sub_dim[0] * sub_dim[1] != len(y) or sub_dim[0] * sub_dim[1] != len(x):
            raise ValueError("sub_dim dimension {} does not match dimension of x {} or y {}".format(sub_dim, len(y),
                                                                                                    len(x)))
        fig, ax = plt.subplots(sub_dim[0], sub_dim[1], figsize=figsize)

        counter = 0
        for i in range(sub_dim[0]):
            for j in range(sub_dim[1]):
                ax[i, j].plot(x[counter], y[counter], label=plot_label[counter])
                ax[i, j].set_xlabel(x_label[counter])
                ax[i, j].set_ylabel(y_label[counter])
                if y_log_scale == True:
                    ax.set_yscale('log')
                ax[i, j].set_title(title[counter])
                ax[i, j].grid(True)
                counter += 1

    plt.tight_layout()



    subdir = "./plot"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    fig.savefig(os.path.join(subdir, filename + ".png"),  transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.show()


def imshow(img, figsize=(30, 30), filename=None):
    """
    Custom modified imshow function.

    :param img: image to plot
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """
    img = torch.clamp(img, 0, 1)
    img = img.to('cpu')
    npimg = img.numpy()
    plt.figsize = figsize
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    if filename is not None:
        subdir = "./plot"
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        plt.savefig(os.path.join(subdir, filename + ".png"),  transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.show()


def plot_reconst(model, loader, latent_dim, device='cpu', num_img=1, grid_row_size=10, figsize=(30, 30), filename=None, conditional=False):
    """
    Plot original images and the reconstructed images by the INN

    :param model: INN use for reconstruction
    :param loader: loader that wraps the train, test or evaluation set
    :param latent_dim: dimension of the latent space
    :param num_img: number of images to plot. Default: 1
    :param grid_row_size: number of images in one row in the grid
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: img: original images
             output: reconstructed images
    """

    img, label = next(iter(loader))

    #model = model.to('cpu')
    model.eval()

    img = img.to(device)

    lat_img = model(img)
    shape = lat_img.shape
    lat_img = lat_img.view(lat_img.size(0), -1)
    if conditional:
        binary_label = lat_img.new_zeros(lat_img.size(0), 10)
        idx = torch.arange(label.size(0), dtype=torch.long)
        binary_label[idx, label] = 1
        lat_img_mod = torch.cat([lat_img[:, :latent_dim], binary_label, lat_img.new_zeros((lat_img[:, latent_dim+10:]).shape)], dim=1)
    else:
        lat_img_mod = torch.cat([lat_img[:, :latent_dim], lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim = 1)

    lat_img_mod = lat_img_mod.view(shape)
    output = model(lat_img_mod, rev=True)

    print("Original Image:")
    imshow(torchvision.utils.make_grid(img[:num_img].detach(), grid_row_size), figsize,
           filename + "_original" if (filename is not None) else None)
    print("Reconstructed Image:")
    imshow(torchvision.utils.make_grid(output[:num_img].detach(), grid_row_size), figsize,
           filename + "_reconstructed" if (filename is not None) else None)

    return img, output


def plot_diff(model, loader, latent_dim, device='cpu', num_img=1, grid_row_size=10, figsize=(30, 30), filename=None, conditional=False):
    """
    Plot original images, reconstructed images by the INN and the difference between those images.

    :param model: INN use for reconstruction
    :param loader: loader that wraps the train, test or evaluation set
    :param latent_dim: dimension of the latent space
    :param num_img: number of images to plot. Default: 1
    :param grid_row_size: number of images in one row in the grid
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    img, reconst_img = plot_reconst(model, loader, latent_dim, device, num_img, grid_row_size, figsize, filename, conditional)

    diff_img = (img - reconst_img + 1) / 2

    print("Difference:")
    imshow(torchvision.utils.make_grid(diff_img[:num_img].detach(), grid_row_size), figsize,
           filename + "_difference" if (filename is not None) else None)


def plot_diff_all(get_model, modelname, num_epoch, loader, latent_dim_lst, device='cpu', num_img=1, grid_row_size=10, figsize=(30, 30), filename=None, conditional=False):
    """
    Plot original images, reconstructed images by the INN and the difference between those images for all latent dimensions given in latent_dim_lst.

    :param model: INN use for reconstruction
    :param loader: loader that wraps the train, test or evaluation set
    :param latent_dim_lst: list of dimensions of the latent space of which plots should be generated
    :param num_img: number of images to plot. Default: 1
    :param grid_row_size: number of images in one row in the grid
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """
    for lat_dim in latent_dim_lst:
        print("Latent Dimension: ", lat_dim)
        try:
            model = fm.load_model('{}_{}_{}'.format(modelname, lat_dim, num_epoch), "{}_bottleneck".format(modelname))
            plot_diff(model, loader, lat_dim, device, num_img, grid_row_size, filename='{}_{}'.format(modelname, lat_dim))
        except:
            model = get_model().to(device)
            model = fm.load_weight(model, '{}_{}_{}'.format(modelname, lat_dim, num_epoch), '{}_bottleneck'.format(modelname))
            plot_diff(model, loader, lat_dim, device, num_img, grid_row_size, filename='com_INN_mnist_{}'.format(lat_dim))


def plot_inter(img1, img2, num_steps=10, grid_row_size=10, figsize=(30, 30), filename=None):
    """
    Plot interpolation between two images.

    :param img1: image 1
    :param img2: image 2
    :param num_steps: number of images to interpolate between image 1 and 2
    :param grid_row_size: number of images in one row in the grid
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    img_lst = []
    for p in np.linspace(0, 1, num=num_steps):
        img_temp = p * img1 + (1 - p) * img2
        img_lst.append(img_temp)

    img = torch.stack(img_lst)

    imshow(torchvision.utils.make_grid(img.detach(), grid_row_size), figsize, filename if (filename is not None) else None)



def plot_inter_latent(loader, model, latent_dim, num_steps=8, num_sample=1, figsize=(30, 30), filename=None):
    """
    Plot latent space interpolation between two images from a data loader. Attention: num_steps * num_sample can not be
    bigger the batch size of the loader (This problem will be solved in the future)

    :param loader: loader that wraps the train, test or evaluation set
    :param model: INN used to project the images into the latent space
    :param latent_dim: dimension of the latent space
    :param num_steps: number of images to interpolate between two images
    :param num_sample: number of images to plot
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    img, label = next(iter(loader))

    model.to('cpu')
    model.eval()

    lat_img = model(img)
    lat_shape = lat_img.shape
    lat_img = lat_img.view(lat_img.size(0), -1)

    lat_img_int = []
    for i in range(int(len(lat_img) / num_steps)):
        if i == (len(lat_img) - 1):
            for p in np.linspace(0, 1, num=num_steps):
                lat_img_int_img = p * lat_img[i].detach().numpy() + (1 - p) * lat_img[0].detach().numpy()
                lat_img_int.append(lat_img_int_img)
        else:
            for p in np.linspace(0, 1, num=num_steps):
                lat_img_int_img = p * lat_img[i].detach().numpy() + (1 - p) * lat_img[i + 1].detach().numpy()
                lat_img_int.append(lat_img_int_img)

    lat_img_int = np.array(lat_img_int)
    lat_img_int = torch.from_numpy(lat_img_int)

    lat_img_mod = torch.cat([lat_img_int[:, :latent_dim], lat_img_int.new_zeros((lat_img_int[:, latent_dim:]).shape)], dim=1)
    lat_img_mod = lat_img_mod.view(lat_shape)

    output = model(lat_img_mod, rev=True)

    counter = 0
    for num in range(num_sample):
        inter_row_lst = []
        for i in range(num_steps):
            inter_row_lst.append(output[counter])
            counter += 1
        inter_row = torch.stack(inter_row_lst)
        imshow(torchvision.utils.make_grid(inter_row.detach(), num_steps), figsize,
               filename + "_interpolation_{}".format(num) if (filename is not None) else None)



def plot_samples(model, latent_dim, input_size, input_shape, num_sample=1, grid_row_size=10, figsize=(30, 30), filename=None):
    """
    Generates samples from learned distribution by sampling prior and decoding.

    :param model: INN used for sampling
    :param latent_dim: dimension of the latent space
    :param input_size: total number of elements in the input of the INN
    :param input_shape: shape of the input for the INN
    :param num_sample: number of samples to generate
    :param grid_row_size: number of images in one row in the grid
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    model.to('cpu')
    model.eval()

    prior_samples = tra.traverse_continous_grid(latent_dim, input_size, None, 0, num_sample, grid_row_size, True)

    if len(input_shape) == 2:
        prior_samples = prior_samples.view(num_sample, input_shape[0], input_shape[1])
    elif len(input_shape) == 3:
        prior_samples = prior_samples.view(num_sample, input_shape[0], input_shape[1], input_shape[2])
    else:
        raise ValueError("input_shape is neither 2- nor 3-dimensional")

    generate = model(prior_samples, rev=True)

    imshow(torchvision.utils.make_grid(generate.detach(), grid_row_size), figsize, filename if (filename is not None) else None)


def plot_latent_traversal_line(model, latent_dim, input_size, input_shape, idx, num_sample=1, figsize=(30, 30), filename=None, dataset=None, conditional_target=None, device='cpu'):
    """
    Generates an image traversal through a latent dimension.

    :param model: INN used for sampling
    :param latent_dim: dimension of the latent space
    :param input_size: total number of elements in the input of the INN
    :param input_shape: shape of the input for the INN
    :param idx: Index of a continuous latent dimension to traverse. If None, no latent is traversed and all latent
    dimensions are randomly sampled or kept fixed.
    :param num_sample: number of samples to generate
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :param dataset: dataset to draw images from for which the latent traversal will be created
    :return: None
    """

    model.to(device)
    model.eval()

    if dataset is not None:
        loader = dl.get_loader(dataset, num_sample)

        img, label = next(iter(loader))

        lat_img = model(img)
        lat_img = lat_img.view(lat_img.size(0), -1)
    else:
        lat_img = None

    latent_samples = tra.traverse_continous_line(latent_dim, input_size, idx, num_sample, False, lat_img, conditional_target=conditional_target)

    if len(input_shape) == 2:
        latent_samples = latent_samples.view(num_sample, input_shape[0], input_shape[1])
    elif len(input_shape) == 3:
        latent_samples = latent_samples.view(num_sample, input_shape[0], input_shape[1], input_shape[2])
    else:
        raise ValueError("input_shape is neither 2- nor 3-dimensional")

    generate = model(latent_samples.to(device), rev=True)

    imshow(torchvision.utils.make_grid(generate.detach(), num_sample), figsize, filename if (filename is not None) else None)


def plot_all_traversals_grid(model, latent_dim, input_size, input_shape, num_sample=1, figsize=(30, 30), filename=None, conditional_target=None, device='cpu'):
    """
     Generates a grid of images for all latent dimensions, where each row corresponds to a traversal along a latent
    dimension.

    :param model: INN used for sampling
    :param latent_dim: dimension of the latent space
    :param input_size: total number of elements in the input of the INN
    :param input_shape: shape of the input for the INN
    :param idx: Index of a continuous latent dimension to traverse. If None, no latent is traversed and all latent
    dimensions are randomly sampled or kept fixed.
    :param num_sample: number of samples to generate
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    model.to(device)
    model.eval()

    if len(input_shape) != 3:
        raise ValueError("input_shape must be 3-dimensional")


    grid = []
    for idx in range(latent_dim):
        latent_samples = tra.traverse_continous_line(latent_dim, input_size, idx, num_sample, False, lat_img=None, conditional_target=conditional_target)

        latent_samples = latent_samples.view(num_sample, input_shape[0], input_shape[1], input_shape[2])

        generate = model(latent_samples.to(device), rev=True)

        grid.append(generate)

    grid = torch.cat(grid)

    imshow(torchvision.utils.make_grid(grid.detach(), num_sample), figsize, filename if (filename is not None) else None)


def plot_latent_traversal_grid(model, latent_dim, input_size, input_shape, idx, axis=0, num_sample=1, grid_row_size=10, figsize=(30, 30), filename=None, idx_2=None):
    """
    Generates a grid of image traversals through two latent dimensions.

    :param model: INN used for sampling
    :param latent_dim: dimension of the latent space
    :param input_size: total number of elements in the input of the INN
    :param input_shape: shape of the input for the INN
    :param idx: Index of a continuous latent dimension to traverse. If None, no latent is traversed and all latent
    dimensions are randomly sampled or kept fixed.
    :param axis: Either 0 for traversal across the rows or 1 for traversal across the columns.
    :param num_sample: number of samples to generate
    :param grid_row_size: number of images in one row in the grid
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    latent_samples = tra.traverse_continous_grid(latent_dim, input_size, idx, axis, num_sample, grid_row_size, idx_2=idx_2)

    if len(input_shape) == 2:
        latent_samples = latent_samples.view(num_sample, input_shape[0], input_shape[1])
    elif len(input_shape) == 3:
        latent_samples = latent_samples.view(num_sample, input_shape[0], input_shape[1], input_shape[2])
    else:
        raise ValueError("input_shape is neither 2- nor 3-dimensional")

    generate = model(latent_samples, rev=True)

    imshow(torchvision.utils.make_grid(generate.detach(), grid_row_size), figsize, filename if (filename is not None) else None)


def plot_all_traversals(model, latent_dim, input_size, input_shape, num_sample=8, figsize=(30, 30), filename=None, conditional_target=None, device='cpu'):
    """
    Generates a grid of images for all latent dimensions, where each row corresponds to a traversal along a latent
    dimension.

    :param model: INN used for sampling
    :param latent_dim: dimension of the latent space
    :param input_size: size of the input for INN
    :param num_sample: Number of samples for each latent traversal
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """

    #latent_samples = []

    for idx in range(latent_dim):
        plot_latent_traversal_line(model, latent_dim, input_size, input_shape, idx, num_sample, figsize, filename, conditional_target=conditional_target, device=device)

    #imshow(torchvision.utils.make_grid(generate.detach(), num_sample), figsize, filename if (filename is not None) else None)

def to_img(x, size):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), size[0], size[1], size[2])
    return x
