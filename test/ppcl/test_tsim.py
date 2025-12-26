from tensorpc.apps.mls import tsim
import numpy as np 

def test_tsim_memory():

    mem = tsim.create_sim_memory_single("wtf", np.arange(8 * 8).reshape(8, 8).astype(np.float32))
    offset_m = tsim.arange(16)
    offset_n = tsim.arange(8)
    pointer = tsim.create_pointer_scalar(tsim.DTypeEnum.float32, 2, memory=mem)
    res = pointer.load()
    print(res)
    
    pointer_block = (pointer - 2) + (offset_m[:, None] * 8 + offset_n[None, :])
    res = pointer_block.load(mask=offset_m[:, None] < 8, other=0)
    print(res.storage.data)
    # print(res.storage.indices)

def _main():
    test_tsim_memory()


if __name__ == "__main__":
    _main()