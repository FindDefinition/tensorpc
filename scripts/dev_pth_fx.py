from torch import nn
from torch.nn import functional as F
import torch.fx
import torch.export

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input):
        if input.shape[1] >= 64:
            return F.relu(self.linear(input))
        else:
            return F.gelu(self.linear(input))


def main():
    mod = MyModule(500, 10)
    gm = torch.export.export(mod, args=(torch.rand(128, 500),))
    print(gm)

if __name__ == "__main__":
    main()