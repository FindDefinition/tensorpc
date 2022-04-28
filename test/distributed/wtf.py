import inspect 
from distflow.core import inspecttools

class A:
    def a(self):
        pass 

    @staticmethod 
    def b(x):
        return x 


if __name__ == "__main__":
    members = inspecttools.get_members_by_type(A, False)
    for k, v in members:
            v_static = inspect.getattr_static(A, k)
            v_sig = inspect.signature(v)
            print(len(v_sig.parameters), k)
