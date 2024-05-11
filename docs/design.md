# TensoRPC Design

## Major Features

* every group of node runs in one process
* every group have a rmq, grpc server and http (websocket) server
* use rmq to send small message, use grpc to send large message (manually, by a decorator)


## Node

A node is a simple class that inherit a very simple base class. Node must have a sync method called "handle_message" which is used for rmq. Users need to use ```@rpc_method``` to indicate a method is exported as rpc.

For large message, tensorpc use grpc to send large message: 
1. the large message is saved in current node, and a small specific message is sent via rmq
2. target node launch grpc stream to receive the large message from current node.

We may also add shared-mem support for large message. we can:
1. query master process for shared mem
2. save data to smem, send
3. target process receive this smem block, then send message to master node to release this smem.

### Node Init

If a node has no dependency during init, the init process is trivial. If dependency exists, we must serialize init process for these nodes.

