# TensoRPC

this project contains two major parts:

* A RPC library that focus on simple usage.

* A python UI framework that need to be used with [frontend](https://finddefinition.github.io/devflow/dock).

This project is still in early stage, and the API may change in the future.

for each newest patch version of devflow, we ensure all patch version of tensorpc can be used if devflow and tensorpc have same major/minor version number.

## usage

1. install tensorpc

2. run backend `python -m tensorpc.serve --port=50051 --http_port=50052`

3. check https://finddefinition.github.io/devflow/dock, check `help` button in top right corner.