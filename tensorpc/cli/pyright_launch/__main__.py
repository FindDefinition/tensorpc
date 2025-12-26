from pyright.langserver import run
import fire
import json


def main():
    try:
        run("--stdio")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
