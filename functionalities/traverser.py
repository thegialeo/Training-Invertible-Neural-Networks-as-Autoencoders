import numpy as np
from scipy import stats
import torch


def traverse_continous_line(latent_dim, input_size, idx, num_sample, sample_prior=False, lat_img=None, conditional_target=None):
    """
    Returns samples from latent space, corresponding to a traversal of a continuous latent variable indicated by idx.

    :param latent_dim: dimension of the latent space
    :param input_size: size of the input for INN
    :param idx: Index of continuous latent dimension to traverse. If None, no latent is traversed and all latent
    dimensions are randomly sampled or kept fixed.
    :param num_sample: number of samples to generate
    :param sample_prior: If False fixes samples in untraversed latent dimensions. If True samples untraversed latent
    dimensions from prior.
    :return: samples
    """

    if lat_img is not None:
        samples = lat_img
    else:
        if sample_prior:
            lat_samples = np.random.normal(size=(num_sample, latent_dim))
            zero_samples = np.zeros(shape=(num_sample, input_size - latent_dim))
            samples = np.concatenate((lat_samples, zero_samples), axis=1)
        elif conditional_target is not None:
            lat_samples = np.zeros(shape=(num_sample, latent_dim))
            binary_target = np.zeros(shape=(num_sample, 10))
            idx = np.arange(num_sample)
            binary_target[idx, conditional_target] = 1
            samples = np.concatenate((lat_samples, binary_target, np.zeros(shape=(num_sample, input_size - (latent_dim+10)))), axis=1)
        else:
            samples = np.zeros(shape=(num_sample, input_size))

    if idx is not None:
        cdf_traversal = np.linspace(0.05, 0.95, num_sample)
        cont_traversal = stats.norm.ppf(cdf_traversal)

        for i in range(num_sample):
            samples[i, idx] = cont_traversal[i]

    return torch.Tensor(samples)


def traverse_discrete_line(latent_dim, input_size, idx, num_sample, disc_lst, sample_prior=False, lat_img=None):
    """
    Returns samples from latent space, corresponding to a traversal of a continuous latent variable indicated by idx.

    :param latent_dim: dimension of the latent space
    :param input_size: size of the input for INN
    :param idx: Index of continuous latent dimension to traverse. If None, no latent is traversed and all latent
    dimensions are randomly sampled or kept fixed.
    :param num_sample: number of samples to generate
    :param sample_prior: If False fixes samples in untraversed latent dimensions. If True samples untraversed latent
    dimensions from prior.
    :return: samples
    """

    if lat_img is not None:
        samples = lat_img
    else:
        if sample_prior:
            lat_samples = np.random.normal(size=(num_sample, latent_dim))
            zero_samples = np.zeros(shape=(num_sample, input_size - latent_dim))
            samples = np.concatenate((lat_samples, zero_samples), axis=1)
        else:
            samples = np.zeros(shape=(num_sample, input_size))

        for i in range(num_sample):
            samples[i, idx] = disc_lst[i]

    return torch.Tensor(samples)


def traverse_continous_grid(latent_dim, input_size, idx, axis, num_sample, grid_row_size, sample_prior=False, idx_2=None):
    """
    Returns samples from latent space, corresponding to a two dimensional traversal of the continuous space.

    :param latent_dim: dimension of the latent space
    :param input_size: size of the input for INN
    :param idx: Index of a continuous latent dimension to traverse. If None, no latent is traversed and all latent
    dimensions are randomly sampled or kept fixed.
    :param axis: Either 0 for traversal across the rows or 1 for traversal across the columns.
    :param num_sample: total number of samples to generate
    :param grid_row_size: number of samples in one row of the grid
    :param sample_prior: If False fixes samples in untraversed latent dimensions. If True samples untraversed latent
    dimensions from prior.
    :return: samples
    """

    if sample_prior:
        lat_samples = np.random.normal(size=(num_sample, latent_dim))
        zero_samples = np.zeros(shape=(num_sample, input_size - latent_dim))
        samples = np.concatenate((lat_samples, zero_samples), axis=1)
    else:
        samples = np.zeros(shape=(num_sample, input_size))

    if idx is not None:
        cdf_traversal = np.linspace(0.05, 0.95, grid_row_size if axis == 0 else num_sample // grid_row_size)
        cont_traversal = stats.norm.ppf(cdf_traversal)

        for i in range(grid_row_size):
            for j in range(num_sample // grid_row_size):
                if axis == 0:
                    samples[i * (num_sample // grid_row_size) + j, idx] = cont_traversal[i]
                    if idx_2 is not None:
                        samples[i * (num_sample // grid_row_size) + j, idx_2] = cont_traversal[j]
                else:
                    samples[i * (num_sample // grid_row_size) + j, idx] = cont_traversal[j]
                    if idx_2 is not None:
                        samples[i * (num_sample // grid_row_size) + j, idx_2] = cont_traversal[i]

    return torch.Tensor(samples)

