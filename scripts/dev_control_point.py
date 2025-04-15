import time
import torch 
import os 
import tqdm 
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from tensorpc.apps.distssh.pth import pth_control_point

def main():
    world_size_str = os.getenv("WORLD_SIZE")
    assert world_size_str is not None 
    world_size = int(world_size_str)
    # init dist group
    init_device_mesh("cuda", (world_size,))

    for j in tqdm.tqdm(list(range(30))):
        time.sleep(1)
        pth_control_point()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()