import time
import tqdm 
import sys 
import os 

def main():
    print("CHILD PID", os.getpid())
    print("HELLOHELLOHELLOHELLOHELLO!")
    print("HELLOHELLOHELLOHELLOHELLO!", file=sys.stderr)

    for i in tqdm.tqdm(range(5000)):
        time.sleep(0.001)
    # raise ValueError("test error")

if __name__ == "__main__":
    main()