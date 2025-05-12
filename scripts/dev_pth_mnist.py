from __future__ import division, print_function

import argparse
import os

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from torch.utils import _pytree as pytree

from tensorpc.apps.distssh.client import TorchDistributedCkptClient, pth_control_point, PerfMonitorClient
from tensorpc.apps.distssh.constants import TENSORPC_ENV_DISTSSH_URL_WITH_PORT
import tensorpc 
def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False

import contextlib
from typing import Any, Dict, List, Optional, Tuple, cast
import dataclasses 
import contextvars
import time as pytime

class PerfContext:
    def __init__(self) -> None:
        import torch 
        self._ns_stack: List[str] = []
        self._measures: Dict[Tuple[str, ...], List[Tuple[torch.cuda.Event,
                                                         torch.cuda.Event]]] = {}
        self._print_pair: List[Tuple[str, float]] = []
        self.perf_result: Dict[Tuple[str, ...], List[float]] = {}
        self.perf_result_timestamps: Dict[Tuple[str, ...], List[tuple[int, int]]] = {}

        self._enable_controlled_by_root: Optional[bool] = None
        self.perf_result_root: float = 0.0
        self._first_event: Optional[torch.cuda.Event] = None
        self._first_cpu_timestamp: int = 0

    def get_last_name_perf_results(self) -> dict[str, list[float]]:
        res: dict[str, list[float]] = {}
        for keys, data in self.perf_result.items():
            assert len(keys) > 0
            key = keys[-1]
            if key not in res:
                res[key] = []
            res[key].extend(data)
        return res

    def get_chrome_trace_events(self, pid: int = 0, tid: int = 0, with_cpu_timestamp: bool = True, ignore_root: bool = False) -> list[dict]:
        events: list[dict] = []
        ts_base = 0
        if with_cpu_timestamp:
            ts_base = self._first_cpu_timestamp
        for keys, data in self.perf_result.items():
            if ignore_root and len(keys) == 1:
                continue
            assert len(keys) > 0
            key = ".".join(keys)
            for i, time in enumerate(data):
                start_ts = self.perf_result_timestamps[keys][i][0]
                end_ts = self.perf_result_timestamps[keys][i][1]
                event = {
                    "name": key,
                    "ph": "X",
                    "ts": start_ts + ts_base,
                    "dur": end_ts - start_ts,
                    "pid": pid,
                    "tid": tid,
                }
                events.append(event)
        return events


PERF_CONTEXT: contextvars.ContextVar[
    Optional[PerfContext]] = contextvars.ContextVar("perf_context",
                                                    default=None)


@contextlib.contextmanager
def __enter_perf_conetxt(perf_ctx: PerfContext):
    token = PERF_CONTEXT.set(perf_ctx)
    try:
        yield perf_ctx
    finally:
        PERF_CONTEXT.reset(token)


@contextlib.contextmanager
def perf_context(name: str,
                 *,
                 stream: Optional[Any] = None,
                 enable: bool = True,
                 print_result: bool = True,
                 control_child_enable: bool = False,
                 child_only: bool = False,
                 sync_start_event: bool = False):
    import torch 

    ctx = PERF_CONTEXT.get()
    enter_null = contextlib.nullcontext()
    is_root = False
    root_key = None
    if stream is None:
        stream = torch.cuda.current_stream()
    if ctx is None:
        if child_only:
            yield None
            return
        ctx = PerfContext()
        if control_child_enable:
            ctx._enable_controlled_by_root = enable
        is_root = True
        root_key = (name, )
        enter_null = __enter_perf_conetxt(ctx)
    if ctx._enable_controlled_by_root is not None:
        enable = ctx._enable_controlled_by_root
    if not enable:
        yield None
        return
    ctx._ns_stack.append(name)
    root_time = 1
    try:
        with enter_null:
            ev_start = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
            ev_stop = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
            if ctx._first_event is None:
                if sync_start_event:
                    stream.synchronize()
                ctx._first_cpu_timestamp = pytime.time_ns()

            ev_start.record(stream)
            yield ctx
            ev_stop.record(stream)
            if ctx._first_event is None:
                ctx._first_event = ev_start
            key = tuple(ctx._ns_stack)
            if key not in ctx._measures:
                ctx._measures[key] = []
            ctx._measures[tuple(ctx._ns_stack)].append((ev_start, ev_stop))

    finally:
        ctx._ns_stack.pop()
    if is_root:
        all_times: Dict[Tuple[str, ...], List[float]] = {}
        all_timestamps: Dict[Tuple[str, ...], List[tuple[int, int]]] = {}
        assert ctx._first_event is not None 
        for key, data in ctx._measures.items():
            for pair in data:
                pair[0].synchronize()
                pair[1].synchronize()
            times = [x[0].elapsed_time(x[1]) for x in data]
            timestamps = [(int(ctx._first_event.elapsed_time(x[0]) * 1e6), int(ctx._first_event.elapsed_time(x[1]) * 1e6)) for x in data]
            all_times[key] = times
            all_timestamps[key] = timestamps
            if key == root_key:
                root_time = times[0]
        ctx.perf_result = all_times
        ctx.perf_result_root = root_time
        ctx.perf_result_timestamps = all_timestamps
        ctx._measures.clear()
        if print_result:
            for key, data in all_times.items():
                time = sum(data, 0)
                spaces = "  " * (len(key) - 1)
                if len(key) > 1:
                    print(
                        f"{spaces}[{key[-1]}@{len(data)}]({(time / root_time) * 100:.3f}%): {time:.4}"
                    )
                else:
                    assert len(key) == 1
                    print(f"[{key[-1]}@{len(data)}]: {time:.4}")

class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    @torch.no_grad()
    def update(self, output, target):
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs):
        distssh_url = os.environ.get(TENSORPC_ENV_DISTSSH_URL_WITH_PORT)
        assert distssh_url is not None 
        with tensorpc.RemoteManager(distssh_url) as robj:
            ckpt_client = TorchDistributedCkptClient(robj, 4, 4, -1)
            # flash_ckpt_stream = torch.cuda.Stream()
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = self.train(epoch, ckpt_client)
                test_loss, test_acc = self.evaluate()
                # ckpt_client.store_major_checkpoint("mnist", epoch, {
                #     "model": self.model.state_dict(),
                #     "optimizer": self.optimizer.state_dict(),
                # })
                print(
                    'Epoch: {}/{},'.format(epoch, epochs),
                    'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                    'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
                )
                print(pth_control_point())

    def train(self, epoch, ckpt_client: TorchDistributedCkptClient):
        self.model.train()
        train_loss = Average()
        train_acc = Accuracy()
        cnt = 0
        monitor = PerfMonitorClient()
        for data, target in self.train_loader:
            monitor.record("data")
            with perf_context("root", print_result=False, sync_start_event=True) as ctx:
                with perf_context("data_to_cuda"):

                    data = data.to(self.device)
                    target = target.to(self.device)
                with tensorpc.dbg.manual_trace_scope("fwdbwd"):
                    with perf_context("fwd"):

                        output = self.model(data)
                    with perf_context("loss"):

                        loss = F.cross_entropy(output, target)

                        self.optimizer.zero_grad()
                    with perf_context("bwd"):
                        loss.backward()
            monitor.extend_external_events(ctx.get_chrome_trace_events(ignore_root=False))
            # monitor.record("fwdbwd")
            # wait for minor checkpoint to be stored to remote shm
            # stream.synchronize()
            # torch.cuda.current_stream().wait_stream(stream)
            # torch.cuda.synchronize()
            # if ckpt_client.has_train_checkpoint("mnist", len(self.train_loader) * epoch + cnt):
            #     state_dict_cur = {
            #         "model": self.model.state_dict(),
            #         "optimizer": self.optimizer.state_dict(),
            #     }
            #     state_dict_cur_tensors = pytree.tree_flatten(state_dict_cur)[0]
            #     state_dict_stored_before = ckpt_client.get_train_checkpoint("mnist", len(self.train_loader) * epoch + cnt)
            #     state_dict_stored_before_tensors = pytree.tree_flatten(state_dict_stored_before)[0]
            #     for i in range(len(state_dict_cur_tensors)):
            #         if isinstance(state_dict_cur_tensors[i], torch.Tensor):
            #             assert torch.allclose(state_dict_cur_tensors[i].cpu(), state_dict_stored_before_tensors[i])
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)
            monitor.record("optim")
            monitor.flush_allgather(step=len(self.train_loader) * epoch + cnt, enable=cnt % 10 == 0, scale=0.05)
            cnt += 1
            # if cnt % 20 == 0:
            #     stream.wait_stream(torch.cuda.current_stream())
            #     ckpt_client.store_minor_checkpoint("mnist", len(self.train_loader) * epoch + cnt, {
            #         "model": self.model.state_dict(),
            #         "optimizer": self.optimizer.state_dict(),
            #     }, stream=stream)

        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            test_loss.update(loss.item(), data.size(0))
            test_acc.update(output, target)

        return test_loss, test_acc


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MNISTDataLoader(data.DataLoader):

    def __init__(self, root, batch_size, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if dist.get_rank() == 0:
            datasets.MNIST(root, train=train, transform=transform, download=True)
        dist.barrier()
        dataset = datasets.MNIST(root, train=train, transform=transform, download=False)
        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(dataset)

        super(MNISTDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )


def run(args, mesh):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = Net()
    if distributed_is_initialized():
        model.to(device)
        if args.no_cuda:
            model = nn.parallel.DistributedDataParallel(model)
        else:
            model = nn.parallel.DistributedDataParallel(model, device_mesh=mesh)

    else:
        model = nn.DataParallel(model)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = MNISTDataLoader(args.root, args.batch_size, train=True)
    test_loader = MNISTDataLoader(args.root, args.batch_size, train=False)

    trainer = Trainer(model, optimizer, train_loader, test_loader, device)
    trainer.fit(args.epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='nccl', help='Name of the backend to use.')
    parser.add_argument('-i',
                        '--init-method',
                        type=str,
                        default='tcp://127.0.0.1:23456',
                        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='build/data')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    world_size = os.environ.get('WORLD_SIZE')
    assert world_size is not None 
    if args.no_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("gloo", rank=local_rank)
        # mesh = init_device_mesh("cpu", (int(world_size),))
        mesh = None
        args.rank = local_rank
        args.world_size = int(world_size)
    else:
        mesh = init_device_mesh("cuda", (int(world_size),))

        args.rank = mesh.get_rank()
        args.world_size = mesh.size()
    print(args)

    run(args, mesh)
    if args.no_cuda:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()