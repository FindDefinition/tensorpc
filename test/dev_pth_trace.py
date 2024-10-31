import torch
import numpy as np
from torch import nn
import torch.profiler as profiler
class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx

def main():
    model = MyModule(500, 10).cuda()
    input = torch.rand(128, 500).cuda()
    mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

    # warm-up
    model(input, mask)
    
    with profiler.profile(with_stack=True, profile_memory=True, execution_trace_observer=(
        profiler.ExecutionTraceObserver().register_callback("./execution_trace.json")
    ),) as prof:
        out, idx = model(input, mask)

if __name__ == "__main__":
    main()
