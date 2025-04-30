## FAQ

### Many Zombie Processes in Docker?

use foreground ssh service such as `/usr/bin/sshd -D` to start your docker container, put service command `python -m tensorpc.apps.distssh.cli ...` in the background.

Example (Bash): 

```bash
{ python -m tensorpc.apps.distssh.cli --rank=... --world_size=... --password=... --workdir=... & } && /usr/sbin/sshd -D
```

### `Check failed: state_ == FAILED` when use control point or flash checkpoint

Grpc fork support in python still have some bugs. you need to set pytorch dataloader from `fork` to `spawn` mode.

e.g. put following code to start of your script:

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn')
```