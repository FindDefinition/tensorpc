from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
class App:
    @mark_create_layout
    def my_layout(self):
        
        canvas = three.Canvas([
            three.OrbitControl().prop(makeDefault=True),
            three.uikit.Fullscreen([
                three.uikit.Container([]).prop(flexGrow=1, margin=32, backgroundColor="green"),
                three.uikit.Container([
                    three.uikit.Content([
                        three.Mesh([
                            three.BoxGeometry(),
                            three.MeshBasicMaterial().prop(color="#f0f0f0"),
                        ]).prop(position=(0.0, 0.0, 0), rotation=(0.3, 0.3, 0.3)),
                    ])
                ]).prop(flexGrow=1, margin=32, backgroundColor="blue"),
            ]).prop(backgroundColor="red", sizeX=8, sizeY=4, flexDirection="row")
        ])
        jv = mui.JsonViewer()

        return mui.HBox([
            canvas.prop(flex=2, shadows=True, localClippingEnabled=True),
            mui.VDivider(),
            mui.HBox([
                jv
            ]).prop(flex=1, overflow="auto"),
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden")
