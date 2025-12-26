from tensorpc.dock import mui, three, plus, appctx, mark_create_layout

class App:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000).prop(position=(0, 0, 5))
        canvas = three.Canvas([
            cam,
            three.LineShape(three.Shape.from_aabb(-1, -1, 1, 1)).prop(
                color="red",
                lineWidth=2,                
                position=(0, 0, 0.1),
            ),
            three.CameraControl().prop(makeDefault=True),
            three.InfiniteGridHelper(5, 50, "gray"),
            three.AmbientLight(intensity=3.14),
            three.PointLight().prop(position=(13, 3, 5),
                                    castShadow=True,
                                    color=0xffffff,
                                    intensity=500),
        ])
        return mui.HBox([
            canvas.prop(flex=2, shadows=True),
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden",
                position="relative")
