import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 2,
        hidden_dims=None,
        input_size: int = 28,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        self.input_size = input_size
        self.hidden_dims = hidden_dims

        # -------------------
        # Encoder
        # -------------------
        modules = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]
            self.flattened_size = enc_out.numel()

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # -------------------
        # Decoder
        # -------------------
        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)
        hidden_dims_rev = hidden_dims[::-1]

        modules = []
        for i in range(len(hidden_dims_rev) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_rev[i], 
                        hidden_dims_rev[i + 1], 
                        kernel_size=3, 
                        stride=2, 
                        padding=1, 
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims_rev[i + 1]),
                    nn.LeakyReLU()
                )
            )

        # Last layer of decoder to output original input_channels
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims_rev[-1], 
                    input_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                ),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, *self.enc_shape)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def reconstruct(self, x):
        x_hat, _, _, _ = self.forward(x)
        return x_hat
    
    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.fc_mu.out_features, device=device)
        return self.decode(z)