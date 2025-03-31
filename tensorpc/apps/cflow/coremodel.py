import tensorpc.core.dataclass_dispatch as dataclasses


@dataclasses.dataclass
class ResourceDesc:
    # -1 means use all
    CPU: int = -1
    Mem: int = -1
    # number of GPU can't be -1 because it shouldn't be shared.
    GPU: int = 0
    GPUMem: int = -1

    def validate(self):
        assert self.GPU >= 0

    def is_request_sufficient(self, req_rc: "ResourceDesc"):
        if self.CPU != -1 and self.CPU < req_rc.CPU:
            return False
        if self.Mem != -1 and self.Mem < req_rc.Mem:
            return False
        if self.GPU != -1 and self.GPU < req_rc.GPU:
            return False
        if self.GPUMem != -1 and self.GPUMem < req_rc.GPUMem:
            return False
        return True

    def get_request_remain_rc(self, req_rc: "ResourceDesc"):
        return ResourceDesc(CPU=self.CPU - req_rc.CPU if self.CPU != -1 else -1,
                            Mem=self.Mem - req_rc.Mem if self.Mem != -1 else -1,
                            GPU=self.GPU - req_rc.GPU if self.GPU != -1 else -1,
                            GPUMem=self.GPUMem - req_rc.GPUMem if self.GPUMem != -1 else -1)

    def add_request_rc(self, req_rc: "ResourceDesc"):
        return ResourceDesc(CPU=self.CPU + req_rc.CPU if self.CPU != -1 else -1,
                            Mem=self.Mem + req_rc.Mem if self.Mem != -1 else -1,
                            GPU=self.GPU + req_rc.GPU if self.GPU != -1 else -1,
                            GPUMem=self.GPUMem + req_rc.GPUMem if self.GPUMem != -1 else -1)
