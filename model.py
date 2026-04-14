import torch.nn as nn

class RidgeletInterpretNet(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.linear = nn.Linear(p * p, p + 1, bias=False)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
    def get_weights_as_images(self):
        weights = self.linear.weight.data  # [p+1, p*p]
        return weights.view(self.p + 1, self.p, self.p)
