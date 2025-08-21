import torch
from torch import nn
from torch.fft import rfft

k = 4  # number of paralel 1D images
device = "cpu"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_1a = nn.Conv1d(1, k, kernel_size=5, stride=1)  # 30 -> 26
        self.conv_1b = nn.Conv1d(k, 4*k, kernel_size=3, stride=1)  # -> 24
        self.conv_1c = nn.Conv1d(4*k, 4*k, kernel_size=2, stride=2)  # -> 12

        self.conv_2a = nn.Conv1d(1, k, kernel_size=9, stride=2)  # 60 -> 26
        self.conv_2b = nn.Conv1d(k, 4*k, kernel_size=3, stride=1)  # -> 24
        self.conv_2c = nn.Conv1d(4*k, 4*k, kernel_size=2, stride=2)  # -> 12

        self.conv_3a = nn.Conv1d(1, k, kernel_size=17, stride=4)  # 120 -> 26
        self.conv_3b = nn.Conv1d(k, 4*k, kernel_size=3, stride=1)  # -> 24
        self.conv_3c = nn.Conv1d(4*k, 4*k, kernel_size=2, stride=2)  # -> 12

        self.conv_4a = nn.Conv1d(1, k, kernel_size=34, stride=8)  # 240 -> 26
        self.conv_4b = nn.Conv1d(k, 4*k, kernel_size=3, stride=1)  # -> 24
        self.conv_4c = nn.Conv1d(4*k, 4*k, kernel_size=2, stride=2)  # -> 12

        self.conv_5a = nn.Conv1d(1, k, kernel_size=34, stride=8)  # 240 -> 26
        self.conv_5b = nn.Conv1d(k, 4*k, kernel_size=3, stride=1)  # -> 24
        self.conv_5c = nn.Conv1d(4*k, 4*k, kernel_size=2, stride=2)  # -> 12

        self.linear = nn.Linear(12 * 4 * k * 5, 12)  # big_linear for model 4
        self.fl = nn.Flatten()
        self.ac = nn.GELU()
        self.sg = nn.Sigmoid()

    def forward(self, x):
        """x: (N, 8192) -> (N, 12)"""
        x = torch.abs(rfft(x))  # if sr = 44100, half it
        x1 = x[:, None, 22:52]  # size 30
        x2 = x[:, None, 43:103]  # size 60
        x3 = x[:, None, 86:206]  # size 120
        x4 = x[:, None, 172:412]  # size 240
        x5 = x[:, None, 344:824:2]  # size 240

        x1 = self.ac(self.conv_1a(x1))
        x1 = self.ac(self.conv_1b(x1))
        x1 = self.ac(self.conv_1c(x1))

        x2 = self.ac(self.conv_2a(x2))
        x2 = self.ac(self.conv_2b(x2))
        x2 = self.ac(self.conv_2c(x2))

        x3 = self.ac(self.conv_3a(x3))
        x3 = self.ac(self.conv_3b(x3))
        x3 = self.ac(self.conv_3c(x3))

        x4 = self.ac(self.conv_4a(x4))
        x4 = self.ac(self.conv_4b(x4))
        x4 = self.ac(self.conv_4c(x4))

        x5 = self.ac(self.conv_5a(x5))
        x5 = self.ac(self.conv_5b(x5))
        x5 = self.ac(self.conv_5c(x5))

        x = self.fl(torch.cat((x1, x2, x3, x4, x5), dim=-1))
        return self.sg(self.linear(x))

# model = Model()
# model.load_state_dict(torch.load("NN_parameters/music_NN_6.pth", map_location=device))
# print(model)
