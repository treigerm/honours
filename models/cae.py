import torch
from torch import nn

from .factory import register_model

MSE_LOSS = "mse"
INTER_LOSS = "inter_class_mse"
INTER_INTRA_LOSS = "inter_intra_loss"

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

@register_model("cae")
class CAE(nn.Module):
    """
    CAE modelled after https://github.com/daniel-munro/imageCCA/blob/master/CAE/CAE_GTEx.py.
    """

    def __init__(self, hidden_dims=1024, loss_name=MSE_LOSS):
        super(CAE, self).__init__()
        self.kernel_size = 5
        self.pool_size = 2
        self.hidden_dims = hidden_dims

        self.loss_name = loss_name

        # Use padding 2 to keep the input output dimensions the same.
        self.encoder = nn.Sequential(                        # In:  (b, 3, 128, 128)
            nn.Conv2d(3, 8, self.kernel_size, padding=2),    # Out: (b, 8, 128, 128)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 8, 64, 64)
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, self.kernel_size, padding=2),   # Out: (b, 16, 64, 64)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 16, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, self.kernel_size, padding=2),  # Out: (b, 32, 32, 32)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 32, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, self.kernel_size, padding=2),  # Out: (b, 64, 16, 16)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 64, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, self.kernel_size, padding=2), # Out: (b, 128, 8, 8)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 128, 4, 4)
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(128*4*4, self.hidden_dims)             # Out: (b, self.hidden_dims)
        )

        self.decoder = nn.Sequential(                                 # In: (b, self.hidden_dims)       
            nn.Linear(self.hidden_dims, 128*4*4),
            nn.ReLU(True),
            Reshape(-1, 128, 4, 4),
            nn.Upsample(size=(8, 8)),                                 # Out: (b, 128, 8, 8)
            nn.ConvTranspose2d(128, 64, self.kernel_size, padding=2), # Out: (b, 64, 8, 8)
            nn.ReLU(True),
            nn.Upsample(size=(16, 16)),                               # Out: (b, 64, 16, 16)
            nn.ConvTranspose2d(64, 32, self.kernel_size, padding=2),  # Out: (b, 32, 16, 16)
            nn.ReLU(True),
            nn.Upsample(size=(32, 32)),                               # Out: (b, 32, 32, 32)
            nn.ConvTranspose2d(32, 16, self.kernel_size, padding=2),  # Out: (b, 16, 32, 32)
            nn.ReLU(True),
            nn.Upsample(size=(64, 64)),                               # Out: (b, 16, 64, 64)
            nn.ConvTranspose2d(16, 8, self.kernel_size, padding=2),   # Out: (b, 8, 64, 64)
            nn.ReLU(True),
            nn.Upsample(size=(128, 128)),                             # Out: (b, 8, 128, 128)
            nn.ConvTranspose2d(8, 3, self.kernel_size, padding=2),    # Out: (b, 3, 128, 128)
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def loss(self, x, y):
        if self.loss_name == MSE_LOSS:
            mse = nn.MSELoss()
            return mse(self.forward(x), x)
        elif self.loss_name == INTER_LOSS:
            return self.inter_loss(x, y)
        elif self.loss_name == INTER_INTRA_LOSS:
            return self.inter_intra_loss(x, y)
    
    def inter_loss(self, x, y):
        mse = nn.MSELoss()
        h = self.encoder(x)
        x_reconstructed = self.decoder(h)
        if torch.sum(y == 1) == 0 or torch.sum(y == 0) == 0:
            # If the batch only contains one class return normal mse loss.
            return mse(x_reconstructed, x)

        mean_0 = torch.mean(h[y == 0], dim=0)
        mean_1 = torch.mean(h[y == 1], dim=0)
        return mse(x_reconstructed, x) - mse(mean_0, mean_1)
    
    def inter_intra_loss(self, x, y):
        mse = nn.MSELoss()
        h = self.encoder(x)
        x_reconstructed = self.decoder(h)
        if torch.sum(y == 1) == 0 or torch.sum(y == 0) == 0:
            # If the batch only contains one class return normal mse loss.
            return mse(x_reconstructed, x)
        
        h_0 = h[y == 0] # (b_0, hidden_dims)
        h_1 = h[y == 1] # (b_1, hidden_dims)
        mean_0 = torch.mean(h_0, dim=0) # (hidden_dims,)
        mean_1 = torch.mean(h_1, dim=0) # (hidden_dims,)

        intra_0 = (mean_0.unsqueeze(0).repeat(h_0.shape[0], 1) - h_0).norm(2, dim=0)
        intra_1 = (mean_1.unsqueeze(0).repeat(h_1.shape[0], 1) - h_1).norm(2, dim=0)
        return mse(x_reconstructed, x) - mse(mean_0, mean_1) + intra_0.mean() + intra_1.mean()

@register_model("test_cae")
class TestCAE(nn.Module):
    """Small CAE model used for testing."""

    def __init__(self, loss_name=MSE_LOSS):
        super(TestCAE, self).__init__()
        self.kernel_size = 4
        self.stride = 4
        self.pool_size = 4
        self.hidden_dims = 16

        self.loss_name = loss_name

        self.encoder = nn.Sequential(                              # In: (b, 3, 128, 128)
            nn.Conv2d(3, 8, self.kernel_size, stride=self.stride), # Out: (b, 8, 32, 32)
            nn.MaxPool2d(self.pool_size),                          # Out: (b, 8, 8, 8)
            nn.ReLU(True),
            Flatten(),
            nn.Linear(8*8*8, self.hidden_dims)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dims, 8*8*8),
            nn.ReLU(True),
            Reshape(-1, 8, 8, 8),
            nn.Upsample(size=(128, 128)),
            nn.ConvTranspose2d(8, 3, 5, padding=2),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def loss(self, x, y):
        if self.loss_name == MSE_LOSS:
            mse = nn.MSELoss()
            return mse(self.forward(x), x)
        elif self.loss_name == INTER_LOSS:
            return self.inter_loss(x, y)
        elif self.loss_name == INTER_INTRA_LOSS:
            return self.inter_intra_loss(x, y)
    
    def inter_loss(self, x, y):
        y.requires_grad_(False)
        mse = nn.MSELoss()
        h = self.encoder(x)
        x_reconstructed = self.decoder(h)
        if torch.sum(y == 1) == 0 or torch.sum(y == 0) == 0:
            # If the batch only contains one class return normal mse loss.
            return mse(x_reconstructed, x)

        mean_0 = torch.mean(h[y == 0], dim=0)
        mean_1 = torch.mean(h[y == 1], dim=0)
        return mse(x_reconstructed, x) - mse(mean_0, mean_1)
    
    def inter_intra_loss(self, x, y):
        mse = nn.MSELoss()
        h = self.encoder(x)
        x_reconstructed = self.decoder(h)
        if torch.sum(y == 1) == 0 or torch.sum(y == 0) == 0:
            # If the batch only contains one class return normal mse loss.
            return mse(x_reconstructed, x)
        
        h_0 = h[y == 0] # (b_0, hidden_dims)
        h_1 = h[y == 1] # (b_1, hidden_dims)
        mean_0 = torch.mean(h_0, dim=0) # (hidden_dims,)
        mean_1 = torch.mean(h_1, dim=0) # (hidden_dims,)

        intra_0 = (mean_0.unsqueeze(0).repeat(h_0.shape[0], 1) - h_0).norm(2, dim=0)
        intra_1 = (mean_1.unsqueeze(0).repeat(h_1.shape[0], 1) - h_1).norm(2, dim=0)
        return mse(x_reconstructed, x) - mse(mean_0, mean_1) + intra_0.mean() + intra_1.mean()