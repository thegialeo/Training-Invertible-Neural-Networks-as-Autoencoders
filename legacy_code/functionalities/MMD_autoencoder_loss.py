import torch
import torch.nn as nn
import torchvision.models as models
from functionalities import loss

class MMD_autoencoder_loss(nn.Module):
    def __init__(self, a_distr, a_rec, a_spar, a_disen=0, a_disc=0, latent_dim=8, loss_type='l1', device='cpu',
                 conditional=False, disc_lst=None, feat_idx=None, cont_min=None, cont_max=None, num_iter=None):
        super(MMD_autoencoder_loss, self).__init__()
        self.a_distr = a_distr
        self.a_rec = a_rec
        self.a_spar = a_spar
        self.a_disen = a_disen
        self.a_disc = a_disc
        self.disc_lst = disc_lst
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.device = device
        self.conditional = conditional
        self.cont_min = cont_min
        self.cont_max = cont_max
        self.num_iter = num_iter
        self.num_step = 0

        if self.loss_type == 'vgg':
            if feat_idx is None:
                feat_idx = 5
            vgg16 = models.vgg16(pretrained=True)
            self.feat_model = nn.Sequential(*list(vgg16.features[:feat_idx]))
            self.feat_model.eval()
            self.feat_model.to(self.device)
        elif self.loss_type == 'resnet':
            if feat_idx is None:
                feat_idx = 4
            resnet18 = models.resnet18(pretrained=True)
            self.feat_model = nn.Sequential(*list(resnet18.children())[:feat_idx])
            self.feat_model.eval()
            self.feat_model.to(self.device)

    def update_num_step(self, num_step):
        self.num_step = num_step

    def forward(self, z, v, z_, label=None, target=None):
        if self.loss_type == 'l1':
            l_rec = self.a_rec * loss.l1_loss(z_, z)
        elif self.loss_type == 'l2':
            l_rec = self.a_rec * loss.l2_loss(z_, z)
        elif self.loss_type == 'vgg' or self.loss_type == 'resnet':
            l_rec = self.a_rec * loss.feat_loss(z_, z, self.feat_model)
        else:
            print('loss not found')


        l_disen = self.a_disen * loss.MMD_multiscale(v[:, :self.latent_dim], loss.shuffle(v[:, :self.latent_dim].to(self.device)), self.device)

        if target is not None and self.conditional:
            y = v.new_empty((v.size(0), self.latent_dim)).normal_()
            v_array = []
            for i in range(10):
                temp = v[torch.eq(label, i)][:, :self.latent_dim]
                count = temp.size(0)
                if count == 0:
                    next
                v_array.append(self.a_distr * loss.MMD_multiscale(temp, y[:count], self.device))
            l_distr = torch.stack(v_array).mean()
            #l_distr = self.a_distr * loss.MMD_multiscale(v[:, :self.latent_dim], y, self.device)
            l_sparse = self.a_spar * torch.mean(v[:, self.latent_dim+10:] ** 2)
            l_disc = self.a_disc * loss.l2_loss(v[:, self.latent_dim : self.latent_dim+10], target)
            l = l_rec.to(self.device) + l_distr.to(self.device) + l_sparse.to(self.device) + l_disen.to(self.device) + l_disc.to(self.device)
            return [l, l_rec, l_distr, l_sparse, l_disen, l_disc]
        elif label is not None and self.disc_lst is not None:
            y = v.new_empty((v.size(0), self.latent_dim - 1)).normal_()
            l_distr = self.a_distr * loss.MMD_multiscale(v[:, 1:self.latent_dim], y, self.device)
            l_sparse = self.a_spar * torch.mean(v[:, self.latent_dim:] ** 2)
            l_disc = self.a_disc * loss.l2_loss(v[:, :1], self.disc_lst[label].float())
            l = l_rec.to(self.device) + l_distr.to(self.device) + l_sparse.to(self.device) + l_disen.to(self.device) + l_disc.to(self.device)
            return [l, l_rec, l_distr, l_sparse, l_disen, l_disc]
        elif self.disc_lst is not None:
            y = v.new_empty((v.size(0), self.latent_dim - 1)).normal_()
            l_distr = self.a_distr * loss.MMD_multiscale(v[:, 1:self.latent_dim], y, self.device)
            l_sparse = self.a_spar * torch.mean(v[:, self.latent_dim:] ** 2)
            l_disc = self.a_disc * torch.mean(torch.min(torch.abs(v[:, :1] - self.disc_lst), 1)[0])
            l = l_rec.to(self.device) + l_distr.to(self.device) + l_sparse.to(self.device) + l_disen.to(self.device) + l_disc.to(self.device)
            return [l, l_rec, l_distr, l_sparse, l_disen, l_disc]
        else:
            y = v.new_empty((v.size(0), self.latent_dim)).normal_()
            if self.cont_min is not None and self.cont_max is not None and self.num_iter is not None:
                cont_capacity = (self.cont_max - self.cont_min) * self.num_step / float(self.num_iter) + self.cont_min
                cont_capacity = min(cont_capacity, self.cont_max)
                l_distr = loss.MMD_multiscale(v[:, :self.latent_dim], y, self.device)
                l_distr = self.a_distr * torch.abs(l_distr - cont_capacity)
            else:
                l_distr = self.a_distr * loss.MMD_multiscale(v[:, :self.latent_dim], y, self.device)
            l_sparse = self.a_spar * torch.mean(v[:, self.latent_dim:] ** 2)
            l = l_rec.to(self.device) + l_distr.to(self.device) + l_sparse.to(self.device) + l_disen.to(self.device)
            return [l, l_rec, l_distr, l_sparse, l_disen]
