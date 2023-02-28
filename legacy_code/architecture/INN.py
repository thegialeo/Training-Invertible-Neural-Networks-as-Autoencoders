from FrEIA import framework as fr
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
from FrEIA.modules import reshapes as re


def mnist_inn_com(mask_size=[28, 28]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of MNIST images
    :return:
    """

    img_dims = [1, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                    'F_args': {'channels_hidden': 100}, 'clamp': 1}, name='conv1')

    conv2 = fr.Node([conv1.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                    'F_args': {'channels_hidden': 100}, 'clamp': 1}, name='conv2')

    conv3 = fr.Node([conv2.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                    'F_args': {'channels_hidden': 100}, 'clamp': 1}, name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (img_dims[0]*img_dims[1]*img_dims[2],)}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_small_connected, 'F_args': {'internal_size': 180}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (4, 14, 14)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, r1, r2, r3, r4, fc]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder




def cifar_inn_com(mask_size=[32, 32]):
    """
    Return CIFAR INN autoencoder for comparison with classical autoencoder (same number of parameters).

    :param latent_dim: dimension of the latent space
    :param mask_size: size of the input. Default: Size of CIFAR images
    :param batch_norm: use batch norm for the F_conv modules
    :return: CIFAR INN autoencoder
    """

    img_dims = [3, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.glow_coupling_layer,
                    {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                    name='conv1')

    conv2 = fr.Node([conv1.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv2')

    conv3 = fr.Node([conv2.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (img_dims[0]*img_dims[1]*img_dims[2],)}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer,
                 {'F_class': fu.F_small_connected, 'F_args': {'internal_size': 1000}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (12, 16, 16)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, fc, r1, r2, r3, r4]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


def celeba_inn_com(mask_size=[156, 128]):
    """
    Return CelebA INN autoencoder for comparison with classical autoencoder (same number of parameters).

    :param latent_dim: dimension of the latent space
    :param mask_size: size of the input. Default: Size of CelebA images
    :param batch_norm: use batch norm for the F_conv modules
    :return: CelebA INN autoencoder
    """

    img_dims = [3, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv11 = fr.Node([r1.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv11')

    conv12 = fr.Node([conv11.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv12')

    conv13 = fr.Node([conv12.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv13')

    r2 = fr.Node([conv13.out0], re.haar_multiplex_layer, {}, name='r2')

    conv21 = fr.Node([r2.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv21')

    conv22 = fr.Node([conv21.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv22')

    conv23 = fr.Node([conv22.out0], la.glow_coupling_layer,
                     {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128}, 'clamp': 1},
                     name='conv23')

    r3 = fr.Node([conv23.out0], re.reshape_layer, {'target_dim': (img_dims[0]*img_dims[1]*img_dims[2],)}, name='r3')

    fc = fr.Node([r3.out0], la.rev_multiplicative_layer,
                 {'F_class': fu.F_small_connected, 'F_args': {'internal_size': 200}, 'clamp': 1}, name='fc')

    r4 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (48, 39, 32)}, name='r4')

    r5 = fr.Node([r4.out0], re.haar_restore_layer, {}, name='r5')

    r6 = fr.Node([r5.out0], re.haar_restore_layer, {}, name='r6')

    outp = fr.OutputNode([r6.out0], name='output')

    nodes = [inp, outp, conv11, conv12, conv13, conv21, conv22, conv23, fc, r1, r2, r3, r4, r5, r6]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder
