from tensorpc.apps.ppcl import tsim
import numpy as np 

def test_tsim_memory():

    mem = tsim.create_sim_memory("wtf", np.arange(8 * 8).reshape(8, 8).astype(np.float32))
    offset_m = tsim.arange(16)
    offset_n = tsim.arange(8)
    pointer = tsim.create_pointer_tensor_scalar(0, tsim.DTypeEnum.float32, memory=mem) + (offset_m[:, None] * 8 + offset_n[None, :])
    res = pointer.load(mask=offset_m[:, None] < 8, other=0)
    print(res.storage.data)
    print(res.storage.indices)

def _main():
    test_tsim_memory()


if __name__ == "__main__":
    _main()