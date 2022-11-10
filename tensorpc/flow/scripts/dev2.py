class A:
    def func(self, a, b):
        return a + b 

b = lambda x: x + 1
print(A.func, dir(A.func), A.func.__qualname__)
print(b.__qualname__)