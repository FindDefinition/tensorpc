import functools
import sys
import os
import tokenize
import ast
from typing import Dict, Tuple

class PythonSourceASTCache:
    """cached read whole python source ast.
    """
    def __init__(self):
        self.cache: Dict[str, Tuple[int, float, str, str, ast.AST]] = {}

    def clearcache(self):
        """Clear the cache entirely."""
        self.cache.clear()

    def getsource(self, filename, module_globals=None):
        """Get the lines for a Python source file from the cache.
        Update the cache if it doesn't contain an entry for this file already."""

        if filename in self.cache:
            entry = self.cache[filename]
            if len(entry) != 1:
                return self.cache[filename][2]

        try:
            return self.updatecache(filename, module_globals)
        except MemoryError:
            self.clearcache()
            return []

    def getast(self, filename, module_globals=None):
        """Get the lines for a Python source file from the cache.
        Update the cache if it doesn't contain an entry for this file already."""

        if filename in self.cache:
            entry = self.cache[filename]
            if len(entry) != 1:
                return self.cache[filename][4]

        try:
            return self.updatecache(filename, module_globals)
        except MemoryError:
            self.clearcache()
            return None


    def checkcache(self, filename=None):
        """Discard cache entries that are out of date.
        (This is not checked upon each call!)"""

        if filename is None:
            filenames = list(self.cache.keys())
        elif filename in self.cache:
            filenames = [filename]
        else:
            return

        for filename in filenames:
            entry = self.cache[filename]
            if len(entry) == 1:
                self.cache.pop(filename)

    def updatecache(self, filename, module_globals=None):
        """Update a cache entry and return its list of lines.
        If something's wrong, delete the cache entry.
        Update the cache if it doesn't contain an entry for this file already."""

        fullname = filename
        try:
            stat = os.stat(fullname)
        except OSError:
            del self.cache[filename]
            return None

        try:
            with tokenize.open(fullname) as fp:
                source = fp.read()
            tree = ast.parse(source)
        except OSError:
            del self.cache[filename]
            return None
        except ValueError:
            del self.cache[filename]
            return None

        size, mtime = stat.st_size, stat.st_mtime
        if source and not source.endswith('\n'):
            source += '\n'

        self.cache[filename] = size, mtime, source, fullname, tree
        return tree