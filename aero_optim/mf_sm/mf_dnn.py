import numpy as np
import os
import torch


class NNL(torch.nn.Module):
    """
    Low fidelity Neural Network.
    """
    def __init__(
        self,
        layer_sizes_low_fidelity: list[int],
        activation: torch.nn.Module = torch.nn.Tanh()
    ):
        super().__init__()
        self.lay_size_lo = layer_sizes_low_fidelity
        self.activation = activation

        # low fidelity
        lo_layers: list[torch.nn.Module] = []
        for idx in range(len(self.lay_size_lo) - 2):
            lo_layers.append(torch.nn.Linear(self.lay_size_lo[idx], self.lay_size_lo[idx + 1]))
            lo_layers.append(self.activation)
        lo_layers.append(torch.nn.Linear(self.lay_size_lo[-2], self.lay_size_lo[-1]))
        self.nn_lo = torch.nn.Sequential(*lo_layers)

    def forward(self, x: torch.Tensor):
        return self.nn_lo(x)


class NNH(torch.nn.Module):
    """
    High fidelity Neural Networks.
    """
    def __init__(
        self,
        layer_sizes_NNH1: list[int],
        layer_sizes_NNH2: list[int],
        activation: torch.nn.Module = torch.nn.Tanh()
    ):
        super().__init__()
        self.lay_size_NNH1 = layer_sizes_NNH1
        self.lay_size_NNH2 = layer_sizes_NNH2
        self.activation = activation

        # high fideliy
        # linear
        NNH1_layers: list[torch.nn.Module] = []
        for idx in range(len(self.lay_size_NNH1) - 1):
            NNH1_layers.append(
                torch.nn.Linear(self.lay_size_NNH1[idx], self.lay_size_NNH1[idx + 1])
            )
        self.NNH1 = torch.nn.Sequential(*NNH1_layers)
        # nonlinear
        NNH2_layers: list[torch.nn.Module] = []
        for idx in range(len(self.lay_size_NNH2) - 2):
            NNH2_layers.append(
                torch.nn.Linear(self.lay_size_NNH2[idx], self.lay_size_NNH2[idx + 1])
            )
            NNH2_layers.append(self.activation)
        NNH2_layers.append(torch.nn.Linear(self.lay_size_NNH2[-2], self.lay_size_NNH2[-1]))
        self.NNH2 = torch.nn.Sequential(*NNH2_layers)

        # linear + nonlinear
        self.alpha = torch.nn.Parameter(
            torch.tensor([0.] * self.lay_size_NNH1[-1], requires_grad=True)
        )

    def forward(self, x: torch.Tensor):
        y_hi_l = self.NNH1(x)
        y_hi_nl = self.NNH2(x)
        alpha = torch.nn.functional.tanh(self.alpha)
        return alpha * y_hi_l + (1 - alpha) * y_hi_nl


def NNL_pretrain(
        model: NNL,
        optimizer: torch.optim.Adam,
        x_lo: np.ndarray, y_lo: np.ndarray,
        loss_target: float,
        niter: int,
        device: torch.device,
        scheduler: None | torch.optim.lr_scheduler.MultiStepLR = None
):
    print("NNL pretraining..")
    # move tensor to GPU
    x_lo_torch = torch.from_numpy(x_lo).float()
    x_lo_torch = x_lo_torch.to(device)
    y_true_lo = torch.from_numpy(y_lo).float()
    y_true_lo = y_true_lo.to(device)
    # start training
    loss_value: float = 1.
    it = 0
    while loss_value > loss_target and it < niter:
        pred_NNL = model(x_lo_torch)
        loss = torch.nn.functional.mse_loss(pred_NNL, y_true_lo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if scheduler:
            scheduler.step()
        if it % 50 == 0:
            print('It:', it, 'Loss', loss.item())
        it = it + 1
    print('It:', it, 'Loss', loss.item())


def MfDNN_train(
    model_NNL: NNL | torch.nn.Module,
    model_NNH: NNH | torch.nn.Module,
    optimizer: torch.optim.Adam,
    x_lo: np.ndarray, y_lo: np.ndarray,
    x_hi: np.ndarray, y_hi: np.ndarray,
    loss_target: float,
    niter: int,
    device: torch.device,
    outdir: str,
    scheduler: None | torch.optim.lr_scheduler.MultiStepLR = None
):
    print("MfDNN training..")
    # move tensors to GPU
    x_lo_torch = torch.from_numpy(x_lo).float()
    x_lo_torch = x_lo_torch.to(device)
    x_hi_torch = torch.from_numpy(x_hi).float()
    x_hi_torch = x_hi_torch.to(device)
    y_true_lo = torch.from_numpy(y_lo).float()
    y_true_lo = y_true_lo.to(device)
    y_true_hi = torch.from_numpy(y_hi).float()
    y_true_hi = y_true_hi.to(device)
    # start training
    loss_value = min_loss = 1.
    it = 0
    while loss_value > loss_target and it < niter:
        pred_NNL_lo = model_NNL(x_lo_torch)
        loss_NNL_lo = torch.nn.functional.mse_loss(pred_NNL_lo, y_true_lo)

        pred_NNL_hi = model_NNL(x_hi_torch)
        pred_NNH = model_NNH(torch.cat((x_hi_torch, pred_NNL_hi), 1))
        loss = torch.nn.functional.mse_loss(pred_NNH, y_true_hi) + loss_NNL_lo

        loss_value = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # print info
        if it % 100 == 0:
            print('It:', it, 'Loss:', loss.item())
        if loss_value < min_loss:
            min_loss = loss_value
            save_model(model_NNL, os.path.join(outdir, "best_NNL.pth"))
            save_model(model_NNH, os.path.join(outdir, "best_NNH.pth"))

        it = it + 1
    print('It:', it, 'Loss', loss.item())
    # load the best model if it exists
    try:
        model_NNL = load_model(os.path.join(outdir, "best_NNL.pth"))
        model_NNH = load_model(os.path.join(outdir, "best_NNH.pth"))
    # the trained model is kept as is otherwise
    except FileNotFoundError:
        save_model(model_NNL, os.path.join(outdir, "best_NNL.pth"))
        save_model(model_NNH, os.path.join(outdir, "best_NNH.pth"))


def save_model(model: torch.nn.Module, path: str):
    torch.save(model, path)


def load_model(path: str) -> torch.nn.Module:
    return torch.load(path, weights_only=False)


def weights_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)
