import fire
from tensorpc.flow.langserv.pyls import serve_ls


import asyncio 
def main(port: int):
    asyncio.run(serve_ls(port=port, ls_cmd=["clangd"]))
if __name__ == "__main__":
    
    fire.Fire(main)