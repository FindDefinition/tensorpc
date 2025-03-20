from tensorpc.dock.components.plus.vis import vapi_core as V
from tensorpc.dock.components.plus.vis.canvas import ComplexCanvas

def main():
    canvas = ComplexCanvas()
    with V.group("haha", canvas=canvas):
        with V.group("wtf.rtx"):
            pass 

        with V.group("wtf.rtx.ccc"):
            pass 

        with V.group("wtf.asd.ccc"):
            pass 


    pass

if __name__ == "__main__":
    main() 