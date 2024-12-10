import torch 
import torch.export 
from tensorpc.flow.components.flowplus.network.pthfx import FlowUIInterpreter, PytorchExportBuilder

def _main():
    from torchvision.models import resnet18
    from torch.nn import functional as F
    from tensorpc.flow import observe_function

    class TestMod(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 7)
            self.bn = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2)
            self.fc = torch.nn.Conv2d(64, 10, 1, 1)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
            # x = torch.flatten(x, 1)
            x = self.fc(x)
            return x.to(torch.float16)

    class VeryVeryVeryVeryVeryLongBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod_list = torch.nn.ModuleList([
                torch.nn.Conv2d(3, 64, 7),
                torch.nn.BatchNorm2d(64),
            ])

        def forward(self, x):
            for m in self.mod_list:
                x = m(x)
            return x

    class TestMod2(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.mod_list = torch.nn.ModuleList([
                VeryVeryVeryVeryVeryLongBlock(),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 10, 1, 1)

            ])
        
        @observe_function
        def forward(self, x):
            for m in self.mod_list:
                x = m(x)
            return x.to(torch.float16)

    r18 = TestMod2()
    # r18 = TestModX()
    # gm = torch.fx.symbolic_trace(m)
    # gm = torch.export.export(m, (torch.rand(8, 4),))
    with torch.device("meta"):
        gm = torch.export.export(r18.to("meta"),
                                 (torch.rand(1, 3, 224, 224), ))
    print(gm)
    import rich
    for node in gm.graph.nodes:
        rich.print(node.name, node.op, node.meta)
    # rich.print(gm.graph_module)
    return
    # rich.print(gm.module())
    # print(gm.graph_module)
    # print(gm.graph_module)
    builder = PytorchExportBuilder()
    interpreter = FlowUIInterpreter(gm,
                                    builder,
                                    original_mod=r18,
                                    verbose=False)
    # inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()
    ftree = builder._build_tree_from_module_stack(builder._id_to_node_data)
    pth_flow = builder.build_pytorch_detached_flow(r18, outputs)
    pth_flow.create_graph_with_expanded_modules(["layer3.1"], r18)
    # rich.print(ftree.root)
    # rich.print(ftree.all_node_ids_with_stack)
    return

    res = [outputs]
    rich.print(outputs)
    graph_res = builder.build_detached_flow(outputs)


def _main_fx():
    from torchvision.models import resnet18

    r18 = resnet18()
    gm = torch.fx.symbolic_trace(r18)
    # gm = torch.export.export(m, (torch.rand(8, 4),))
    import rich
    for node in gm.graph.nodes:
        rich.print(node.meta)
    # print(gm.graph_module)
    # print(gm.graph_module)
    builder = PytorchExportBuilder()
    interpreter = FlowUIInterpreter(gm,
                                    builder,
                                    original_mod=r18,
                                    verbose=True)
    # inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()
    res = [outputs]
    rich.print(outputs)
    # graph_res = builder.build_detached_flow(outputs)


if __name__ == "__main__":
    _main()
