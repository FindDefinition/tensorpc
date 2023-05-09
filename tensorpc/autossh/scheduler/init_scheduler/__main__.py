from tensorpc.autossh.scheduler.tmux import get_tmux_scheduler_info_may_create

def main():
    port, _ = get_tmux_scheduler_info_may_create()
    print(port)

if __name__ == "__main__":
    main()