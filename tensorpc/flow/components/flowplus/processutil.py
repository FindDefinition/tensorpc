from concurrent.futures import ProcessPoolExecutor as _SysProcessPoolExecutor




class ProcessPoolExecutor(_SysProcessPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, fn, *args, **kwargs):
        return super().submit(fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return super().map(fn, *iterables, timeout=timeout, chunksize=chunksize)

