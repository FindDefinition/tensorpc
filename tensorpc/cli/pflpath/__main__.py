from tensorpc.core import pfl 
import json 
import fire

def main(expr: str, indent: int = 2):
    node = pfl.compile_pflpath(expr, )
    if indent > 0:
        print(json.dumps(pfl.dump_pflpath(node), separators=(',', ':'), indent=indent))
    else:
        print(json.dumps(pfl.dump_pflpath(node), separators=(',', ':')))
if __name__ == "__main__":
    fire.Fire(main)