from tensorpc.autossh.core import run_ssh_rpc_call
import fire 


def main(arg_event_id: str, ret_event_id: str, rf_port: int):
    run_ssh_rpc_call(arg_event_id, ret_event_id, rf_port)

if __name__ == "__main__":
    fire.Fire(main)