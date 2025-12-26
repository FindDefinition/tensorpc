import contextlib 
import linecache 

@contextlib.contextmanager
def tempfile_in_linecache(path: str, code: str):
    if path not in linecache.cache:
        linecache.cache[path] = (len(code), None, code.splitlines(keepends=True), path)
        try:
            yield 
        finally:
            del linecache.cache[path]
    else:
        prev = linecache.cache[path]
        linecache.cache[path] = (len(code), None, code.splitlines(keepends=True), path)
        try:
            yield 
        finally:
            linecache.cache[path] = prev
