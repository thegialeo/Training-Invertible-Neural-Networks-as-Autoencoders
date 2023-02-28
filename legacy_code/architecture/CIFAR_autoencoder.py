from torch import nn

class cifar_autoencoder(nn.Module):
    def __init__(self, bottleneck=32):
        super(cifar_autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Conv2d(3, 64, 3, stride=1, padding=0)  # b, 64, 30, 30
        self.e2 = nn.Conv2d(64, 128, 3, stride=1, padding=0)  # b, 128, 28, 28
        self.e3 = nn.MaxPool2d(2, stride=2, padding=0)  # b, 128, 14, 14
        self.e4 = nn.Conv2d(128, 256, 3, stride=1, padding=0)  # b, 256, 12, 12
        self.e5 = nn.Conv2d(256, 512, 3, stride=1, padding=0) # b, 512, 10, 10
        self.e6 = nn.MaxPool2d(2, stride=2, padding=0)  # b, 512, 5, 5
        self.e7 = nn.Conv2d(512, 1024, 3, stride=1, padding=0)  # b, 1024, 3, 3
        self.e8 = nn.Linear(1024*3*3, bottleneck)  # bottleneck

        # decoder
        self.d1 = nn.Linear(bottleneck, 1024*3*3)  # b, 1024, 3, 3
        self.d2 = nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=0) # 512, 5, 5
        self.d3 = nn.Upsample(scale_factor=2) # 512, 10, 10
        self.d4 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0)  # b, 256, 12, 12
        self.d5 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0)  # b, 128, 14, 14
        self.d6 = nn.Upsample(scale_factor=2) # 128, 28, 28
        self.d7 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0)  # b, 64, 30, 30
        self.d8 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=0) # b, 3, 32, 32

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def encode(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e2(h1))
        h3 = self.relu(self.e4(self.e3(h2)))
        h4 = self.relu(self.e5(h3))
        h5 = self.relu(self.e7(self.e6(h4)))
        h5 = h5.view(-1, 1024*3*3)
        h6 = self.e8(h5)

        return  h6


    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 1024, 3, 3)
        h2 = self.relu(self.d2(h1))
        h3 = self.relu(self.d4(self.d3(h2)))
        h4 = self.relu(self.d5(h3))
        h5 = self.relu(self.d7(self.d6(h4)))
        h6 = self.tanh(self.d8(h5))

        return h6


    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)

        return x_
