## FAQ

### Many Zombie Processes in Docker?

use foreground ssh service such as `/usr/bin/sshd -D` to start your docker container, put service command `python -m tensorpc.apps.distssh.cli ...` in the background.

Example (Bash): 

```bash
{ python -m tensorpc.apps.distssh.cli --rank=... --world_size=... --password=... --workdir=... & } && /usr/sbin/sshd -D
```