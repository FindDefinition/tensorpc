from tensorpc.autossh.scheduler.tmux import get_tmux_scheduler_info_may_create

def main():
    port, uid = get_tmux_scheduler_info_may_create()
    print(f"{port},{uid}")

if __name__ == "__main__":
    main()