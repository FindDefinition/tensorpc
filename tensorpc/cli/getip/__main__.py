import fire
from tensorpc.utils.wait_tools import get_primary_ip


def main():
    ip = get_primary_ip()
    print(ip)

if __name__ == "__main__":
    fire.Fire(main)