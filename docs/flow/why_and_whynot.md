## Why Not tensorpc.dock

* tensorpc.dock isn't the simplest python UI library. The simplest one is streamlit, it's recommand for beginners and small-project developers.

* tensorpc.dock isn't compatible with jupyter. 

* tensorpc.dock isn't simple for small projects.

* tensorpc.dock requires some html/css knowledge.

users still need to learn basic css and flex layouts.

## Why tensorpc.dock

* tensorpc.dock is designed for large projects.

tensorpc is a class-based UI library. The goal of tensorpc.dock is __insert layout code anywhere without introduce hard dependency__.

users can insert any layout code to existed python class without introduce hard dependency.

all layout functions and callbacks are automatically-reloadable, when you change callbacks or layout functions, they will be reloaded.

If you use streamlit, you will find that your ui code is tightly coupled with your code. it's very hard to create no-ui reuseable code when use streamlit. so streamlit can't deal with large project.

* tensorpc.dock support large data and array data tranfers natively.

* tensorpc.dock support rich event such as pixel-event for images and drag-and-drop. streamlit don't support arbitrary element drag-and-drop.

* tensorpc.dock support 3D natively. streamlit support 3ds via components and has poor performance.

