import torch

from dataclasses import dataclass
from types import FunctionType


def build_mlp_layers(
        input_dim: int,
        inner_dim: list[int],
        output_dim: int,
        activation: torch.nn.Module,
        batch_norm: bool = False
) -> list[torch.nn.Module]:
    """
    Builds generic MLP layers from input to output without activation function for the last layer.
    """
    layers: list[torch.nn.Module] = []
    # input layer
    layers.append(torch.nn.Linear(input_dim, inner_dim[0]))
    layers.append(activation)
    # hidden layer
    for idx in range(len(inner_dim) - 1):
        layers.append(torch.nn.Linear(inner_dim[idx], inner_dim[idx + 1]))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(inner_dim[idx + 1]))
        layers.append(activation)
    # output_layer
    layers.append(torch.nn.Linear(inner_dim[-1], output_dim))
    return layers


@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.

    **Input/Inner**

    - z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
    - z_sample (torch.Tensor): The sampled value of the latent variable z.
    - x_recon (torch.Tensor): The reconstructed output from the VAE.
    - loss (torch.Tensor): The overall loss of the VAE.
    - loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
    - loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor

    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class MLPEncoder(torch.nn.Module):
    def __init__(
            self, input_dim: int, inner_dim: list[int], latent_dim: int,
            activation: torch.nn.Module,
            encoder: bool = True,
            batch_norm: bool = False
    ):
        super().__init__()
        latent_dim = 2 * latent_dim if encoder else latent_dim
        layers = build_mlp_layers(input_dim, inner_dim, latent_dim, activation, batch_norm)
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPDecoder(MLPEncoder):
    def __init__(
            self, input_dim: int, inner_dim: list[int], latent_dim: int,
            activation: torch.nn.ReLU | torch.nn.SiLU,
            batch_norm: bool = False
    ):
        super().__init__(
            latent_dim, inner_dim[::-1],
            input_dim,
            activation,
            encoder=False,
            batch_norm=batch_norm
        )


class CNNEncoder(torch.nn.Module):
    def __init__(
            self, input_dim, latent_dim: int, activation: FunctionType,
            padding: int = 1, dilatation: int = 1, stride: int = 1, kernel_size: int = 3
    ):
        super().__init__()
        self.activation = activation
        # convolution layers
        self.cnn1 = torch.nn.Conv1d(
            in_channels=1, out_channels=16,
            padding=padding, dilation=dilatation, stride=stride, kernel_size=kernel_size
        )
        cnn1_outdim = int(
            (input_dim + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride + 1
        )
        self.cnn2 = torch.nn.Conv1d(
            in_channels=16, out_channels=8,
            padding=padding, dilation=dilatation, stride=stride, kernel_size=kernel_size
        )
        cnn2_outdim = int(
            (cnn1_outdim + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride + 1
        )
        # convolution layers
        self.lin_dim = cnn2_outdim * 8
        self.fc1 = torch.nn.Linear(self.lin_dim, self.lin_dim // 2)
        self.fc2 = torch.nn.Linear(self.lin_dim // 2, self.lin_dim // 4)
        self.fc3 = torch.nn.Linear(self.lin_dim // 4, latent_dim * 2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.cnn1(x)
        x = self.activation(x)

        x = self.cnn2(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        return x


class CNNDecoder(torch.nn.Module):
    def __init__(
            self, lin_dim, latent_dim: int, activation: FunctionType,
            padding: int = 1, dilatation: int = 1, stride: int = 1, kernel_size: int = 3
    ):
        super().__init__()
        self.lin_dim = lin_dim
        self.activation = activation
        # linear layers
        self.fc3 = torch.nn.Linear(latent_dim, lin_dim // 4)
        self.fc2 = torch.nn.Linear(lin_dim // 4, lin_dim // 2)
        self.fc1 = torch.nn.Linear(lin_dim // 2, lin_dim)
        # reverse convolution layers
        self.cnn2 = torch.nn.ConvTranspose1d(
            in_channels=8, out_channels=16,
            padding=padding, dilation=dilatation, stride=stride, kernel_size=kernel_size
        )
        self.cnn1 = torch.nn.ConvTranspose1d(
            in_channels=16, out_channels=1,
            padding=padding, dilation=dilatation, stride=stride, kernel_size=kernel_size
        )

    def forward(self, x):
        x = self.fc3(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = x.reshape(-1, 8, self.lin_dim // 8)

        x = self.cnn2(x)
        x = self.activation(x)

        x = self.cnn1(x)
        return torch.squeeze(x)


class VAE(torch.nn.Module):
    """
    VAE class adapted from:
    https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/

    Note: intersting inputs about VAEs are also available here:
    https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, w_kl: float):
        super().__init__()
        self.w_kl = w_kl
        self.encoder = encoder
        self.softplus = torch.nn.Softplus()
        self.decoder = decoder

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        """
        return dist.rsample()

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        """
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=torch.empty(0),
                loss_recon=torch.empty(0),
                loss_kl=torch.empty(0),
            )

        # compute loss terms
        loss_recon = torch.nn.functional.mse_loss(recon_x, x)
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(
                z.shape[-1], device=z.device
            ).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + self.w_kl * loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )


class VAETrainer:
    """
    VAE trainer class.
    """
    def __init__(
            self,
            model: VAE,
            optimizer: torch.optim.AdamW
    ):
        self.model = model
        self.optimizer = optimizer

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.errors: list[float] = []
        self.lrs: list[float] = []

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int
    ):
        for _ in range(epochs):
            # train
            for _, batch_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                x = batch_data[0]
                x.to(self.device)
                output = self.model(x)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(loss.item())
                self.lrs.append(self.optimizer.param_groups[0]["lr"])

            # val
            val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for valid_batch, batch_data in enumerate(val_loader):
                    x = batch_data[0]
                    x.to(self.device)
                    # model evaluation
                    output = self.model(x)
                    loss = output.loss
                    val_loss += loss.item()
            self.val_losses.append(val_loss / (valid_batch + 1))
