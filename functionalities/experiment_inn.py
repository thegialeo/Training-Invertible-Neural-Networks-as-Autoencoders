from functionalities import dataloader as dl

class experiment_inn:
    def __init__(self, get_model, modelname, num_epoch, batch_size, dataset, latent_dim, lr_init, l2_reg, milestones,
                 number_dev=0):

        if dataset == 'mnist':
            trainset, test_set, classes = dl.load_mnist()
            trainloa