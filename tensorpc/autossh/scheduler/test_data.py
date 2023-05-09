
import time 
from tensorpc.autossh.scheduler.task_client import TaskClient
def simple_task():
    for i in range(5):
        time.sleep(1)
        print("some_task", i)

def simple_task_with_client():
    client = TaskClient()
    for i in range(5):
        if client.check_need_cancel():
            print("cancelled in remote")
            break
        time.sleep(1)
        client.update_task(i / 5)
        print("some_task", i)

if __name__ == "__main__":
    simple_task_with_client()