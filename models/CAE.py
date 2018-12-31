from torch import nn

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CAE(nn.Module):
    """
    CAE modelled after https://github.com/daniel-munro/imageCCA/blob/master/CAE/CAE_GTEx.py.

    TODO: They do not seem to lower the dimensionality of the images, is this correct?
    """

    def __init__(self):
        super(CAE, self).__init__()
        self.kernel_size = 5
        self.pool_size = 2
        self.hidden_dims = 1024
    
    def encode(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, 3, 128, 128)
        """
        # TODO: Input must be (batch_size, channels, 128, 128)
        # Use padding 2 to keep the input output dimensions the same.
        encoder = nn.Sequential(
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
        return encoder(x)
    
    def decode(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, self.hidden_dims)
        """
        decoder = nn.Sequential(                   
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
        return decoder(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))