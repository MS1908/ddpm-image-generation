import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / np.sqrt(2)
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class DownPath(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownPath, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels, out_channels), nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


class UpPath(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpPath, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedLayer(nn.Module):

    def __init__(self, input_dim, emb_dim):
        super(EmbedLayer, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):

    def __init__(self, in_channels, n_features=256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_features = n_features
        self.n_classes = n_classes

        self.init_conv = ResBlock(in_channels, n_features, is_res=True)

        self.down1 = DownPath(n_features, n_features)
        self.down2 = DownPath(n_features, 2 * n_features)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.time_embed1 = EmbedLayer(1, 2 * n_features)
        self.time_embed2 = EmbedLayer(1, n_features)

        self.context_embed1 = EmbedLayer(n_classes, 2 * n_features)
        self.context_embed2 = EmbedLayer(n_classes, n_features)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_features, 2 * n_features, 7, 7),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU()
        )
        self.up1 = UpPath(4 * n_features, n_features)
        self.up2 = UpPath(2 * n_features, n_features)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_features, n_features, 3, 1, 1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden_vec = self.to_vec(down2)

        # convert context to one hot embedding
        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
        c = c * context_mask

        # embed context
        c_emb1 = self.context_embed1(c).view(-1, self.n_features * 2, 1, 1)
        c_emb2 = self.context_embed2(c).view(-1, self.n_features, 1, 1)

        # embed time step
        t_emb1 = self.time_embed1(t).view(-1, self.n_features * 2, 1, 1)
        t_emb2 = self.time_embed2(t).view(-1, self.n_features, 1, 1)

        # Adaptive Group Normalization
        up1 = self.up0(hidden_vec)
        up2 = self.up1(c_emb1 * up1 + t_emb1, down2)  # add and multiply embeddings
        up3 = self.up2(c_emb2 * up2 + t_emb2, down1)

        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alpha_bar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_ab = torch.sqrt(alpha_bar_t)
    one_over_sqrta = 1 / torch.sqrt(alpha_t)

    sqrt_mab = torch.sqrt(1 - alpha_bar_t)
    mab_over_sqrt_mab_inv = (1 - alpha_t) / sqrt_mab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "one_over_sqrta": one_over_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alpha_bar_t": alpha_bar_t,  # \bar{\alpha_t}
        "sqrt_ab": sqrt_ab,  # \sqrt{\bar{\alpha_t}}
        "sqrt_mab": sqrt_mab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrt_mab": mab_over_sqrt_mab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):

    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrt_ab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        # this method is used in training, so samples t and noise randomly
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrt_ab[_ts, None, None, None] * x + self.sqrt_mab[_ts, None, None, None] * noise
        # This is the x_t, which is sqrt(alpha_bar) x_0 + sqrt(1-alpha_bar) * eps
        # We predict the "error term" from this x_t, and return the loss

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0):
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0, 10).to(device)  # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        for i in range(self.n_T, 0, -1):
            # print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = self.one_over_sqrta[i] * (x_i - eps * self.mab_over_sqrt_mab[i]) + self.sqrt_beta_t[i] * z

        return x_i
