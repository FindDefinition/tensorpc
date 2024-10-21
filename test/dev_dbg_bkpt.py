import tensorpc 
from tensorpc.flow import mui
import time 
import numpy as np 
def main(c = 5):
    a = 5
    b = 3
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])
    tensorpc.dbg.breakpoint(name="WTF", init_port=54321)
    time.sleep(2)
    tensorpc.dbg.breakpoint(name="WTF", init_port=54321)

    print("Finish!")

if __name__ == "__main__":
    main()