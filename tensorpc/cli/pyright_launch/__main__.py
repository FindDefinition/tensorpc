from pyright.langserver import run 
import fire
import json
def main():
    run("--stdio")

if __name__ == "__main__":
    fire.Fire(main)