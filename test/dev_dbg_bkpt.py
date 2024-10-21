import tensorpc 
from tensorpc.flow import mui
import time 
def main():
    a = 5
    b = 3
    complex_obj = mui.Button("Hello")
    tensorpc.dbg.breakpoint(name="WTF", init_port=54321)
    time.sleep(2)
    tensorpc.dbg.breakpoint(name="WTF", init_port=54321)

    print("Finish!")

if __name__ == "__main__":
    main()