from typing import List
from tensorpc.utils.wait_tools import get_free_ports

import fire


def main(count: int):
    print(",".join(map(str, get_free_ports(count))))


if __name__ == "__main__":
    fire.Fire(main)
