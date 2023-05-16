import fire
from tensorpc.flow.langserv import get_tmux_lang_server_info_may_create

def main(ls_type: str, uid: str, port: int):
    port = get_tmux_lang_server_info_may_create(ls_type, uid, port)
    print(f"{port}")

if __name__ == "__main__":
    fire.Fire(main)