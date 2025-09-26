import asyncio
import time
import torch
from tensorpc.dock import (App, EditableApp, EditableLayoutApp, leaflet,
                           mark_autorun, mark_create_layout, marker, mui,
                           chart, plus, three, UserObjTree, appctx, V)
import torch.distributed as dist

class TorchDistributedApp:

    @marker.mark_create_layout
    def my_layout(self):
        rank = dist.get_rank()
        self.md = mui.Markdown()
        return mui.VBox([
            mui.Markdown(f"## Torch Distributed Example: Rank {rank}"),
            mui.Button("Dist test", self.on_dist_test),
            self.md,
        ]).prop(width="100%", height="100%", overflow="hidden")

    async def on_dist_test(self):
        # raise NotImplementedError
        print("rank", dist.get_rank(), "world size", dist.get_world_size())
        ten = torch.rand(2, 3).cuda()
        ten_all_gather = torch.empty(dist.get_world_size(), 2, 3).cuda()
        dist.all_gather_into_tensor(ten_all_gather, ten)
        time.sleep(1.0)
        # await asyncio.sleep(1.0)
        print("rank", dist.get_rank(), "after all gather", ten_all_gather)
        await self.md.write(f"""
### Dist all gather result
```
{ten_all_gather.cpu().numpy()}
```
        """)