import tensorpc 

def main():
    a = 5
    b = 3
    tensorpc.dbg.breakpoint(name="WTF", init_port=54321)
    print("Finish!")

if __name__ == "__main__":
    main()