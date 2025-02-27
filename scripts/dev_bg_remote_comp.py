
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.flow.serv_names import serv_names
from tensorpc.flow import mui
from tensorpc.examples.tutorials import MarkdownTutorialsTree
import asyncio 
import time 

    

def main():
    BACKGROUND_SERVER.start_async(port=52051)
    md = mui.Markdown("Hello Remote Component!")
    container = mui.HBox([])
    BACKGROUND_SERVER.execute_service(serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT, 
        "dev", mui.HBox([
            container,
            mui.Button("Click me", lambda : container.set_new_layout({
                "wtf": mui.Markdown(f"Hello Remote Component! time={time.time_ns()}")
            }))
        ]).prop(border="1px solid red"))
    BACKGROUND_SERVER.execute_service(serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT, 
        "complex_dev", MarkdownTutorialsTree())

    try:
        while True:
            time.sleep(100)
    finally:
        BACKGROUND_SERVER.stop()


if __name__ == "__main__":
    main()