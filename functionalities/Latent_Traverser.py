import numpy as np
import torch
from scipy import stats



class Latent_Traverser():
    """
    The Latent_Traverser class is used to generate traversals of the latent space.

    """

    def __init__(self, latent_dim, input_shape):
        """
        Initialize the Latent_Traverser class.

        Parameters
        ----------
        latent_dim: dimension of the latent space
        input_shape: (batch size, shape of input image)
        """
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.sample_prior = False  # If False fixes samples in untraversed
                                   # latent dimensions. If True samples
                                   # untraversed latent dimensions from prior.


    def traverse_line(self, cont_idx=None, disc_idx=None, size=5):
        """
        Returns a (size, D) latent sample, corresponding to a traversal of the
        latent variable indicated by cont_idx or disc_idx.

        Parameters
        ----------
        cont_idx : int or None
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and cont_idx = 7, then the 7th dimension
            will be traversed while all others will either be fixed or randomly
            sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        size : int
            Number of samples to generate.
        """

        samples = []

        samples.append(self.traverse_continuous_line(idx=cont_idx, size=size))

        return torch.cat(samples, dim=1)


    def traverse_continuous_line(self, idx, size):
        """
        Returns a (size, latent_dim) latent sample, corresponding to a traversal
        of a continuous latent variable indicated by idx.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        size : int
            Number of samples to generate.
        """
        if self.sample_prior:
            samples = torch.cat(np.random.normal(size=(size, self.latent_dim)), np.zeros(shape=(size, self.input_shape[])))
        else:
            samples = np.zeros(shape=(size, self.latent_dim))

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            cdf_traversal = np.linspace(0.05, 0.95, size)
            cont_traversal = stats.norm.ppf(cdf_traversal)

            for i in range(size):
                samples[i, idx] = cont_traversal[i]

        return torch.Tensor(samples)