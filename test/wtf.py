from tensorpc.flow import flowui, jsonlike, plus
from tensorpc.flow.components.plus.config import parse_to_control_nodes


def main():
    elk = flowui.ElkLayoutOptions(spacing=flowui.ElkSpacing(nodeNodeBetweenLayers=25))
    elk.considerModelOrder = flowui.ElkConsiderModelOrder(components="NONE")
    
    node = parse_to_control_nodes(elk, elk, "", {})
    print(node) 


if __name__ == '__main__':
    main()