import bisect
import dataclasses
from difflib import SequenceMatcher
import functools
import sys
import os
import tokenize
import ast
from typing import Dict, List, Optional, Tuple
import linecache 


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

class LineCache:
    # impl a class based python package linecache

    def __init__(self):
        self.cache = {}


    def clearcache(self):
        """Clear the cache entirely."""
        self.cache.clear()


    def getline(self, filename, lineno, module_globals=None):
        """Get a line for a Python source file from the cache.
        Update the cache if it doesn't contain an entry for this file already."""

        lines = self.getlines(filename, module_globals)
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1]
        return ''


    def getlines(self, filename, module_globals=None):
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
                # lazy cache entry, leave it lazy.
                continue
            size, mtime, lines, fullname = entry
            if mtime is None:
                continue   # no-op for files loaded via a __loader__
            try:
                stat = os.stat(fullname)
            except OSError:
                self.cache.pop(filename, None)
                continue
            if size != stat.st_size or mtime != stat.st_mtime:
                self.cache.pop(filename, None)


    def updatecache(self, filename, module_globals=None):
        """Update a cache entry and return its list of lines.
        If something's wrong, print a message, discard the cache entry,
        and return an empty list."""

        if filename in self.cache:
            if len(self.cache[filename]) != 1:
                self.cache.pop(filename, None)
        if not filename or (filename.startswith('<') and filename.endswith('>')):
            return []

        fullname = filename
        try:
            stat = os.stat(fullname)
        except OSError:
            basename = filename

            # Realise a lazy loader based lookup if there is one
            # otherwise try to lookup right now.
            if self.lazycache(filename, module_globals):
                try:
                    data = self.cache[filename][0]()
                except (ImportError, OSError):
                    pass
                else:
                    if data is None:
                        # No luck, the PEP302 loader cannot find the source
                        # for this module.
                        return []
                    self.cache[filename] = (
                        len(data),
                        None,
                        [line + '\n' for line in data.splitlines()],
                        fullname
                    )
                    return self.cache[filename][2]

            # Try looking through the module search path, which is only useful
            # when handling a relative filename.
            if os.path.isabs(filename):
                return []

            for dirname in sys.path:
                try:
                    fullname = os.path.join(dirname, basename)
                except (TypeError, AttributeError):
                    # Not sufficiently string-like to do anything useful with.
                    continue
                try:
                    stat = os.stat(fullname)
                    break
                except OSError:
                    pass
            else:
                return []
        try:
            with tokenize.open(fullname) as fp:
                lines = fp.readlines()
        except (OSError, UnicodeDecodeError, SyntaxError):
            return []
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        size, mtime = stat.st_size, stat.st_mtime
        self.cache[filename] = size, mtime, lines, fullname
        return lines


    def lazycache(self, filename, module_globals):
        """Seed the cache for filename with module_globals.

        The module loader will be asked for the source only when getlines is
        called, not immediately.

        If there is an entry in the cache already, it is not altered.

        :return: True if a lazy load is registered in the cache,
            otherwise False. To register such a load a module loader with a
            get_source method must be found, the filename must be a cacheable
            filename, and the filename must not be already cached.
        """
        if filename in self.cache:
            if len(self.cache[filename]) == 1:
                return True
            else:
                return False
        if not filename or (filename.startswith('<') and filename.endswith('>')):
            return False
        # Try for a __loader__, if available
        if module_globals and '__name__' in module_globals:
            name = module_globals['__name__']
            if (loader := module_globals.get('__loader__')) is None:
                if spec := module_globals.get('__spec__'):
                    try:
                        loader = spec.loader
                    except AttributeError:
                        pass
            get_source = getattr(loader, 'get_source', None)

            if name and get_source:
                get_lines = functools.partial(get_source, name)
                self.cache[filename] = (get_lines,)
                return True
        return False


@dataclasses.dataclass
class SCDItem:
    size: int
    mtime: Optional[float]
    lines: List[str]
    diff_opcodes_equal: Optional[List[Tuple[int, int, int, int]]]
    fullname: str

class SourceChangeDiffCache:

    def __init__(self):
        self.cache: Dict[str, SCDItem] = {}
        self.original_lines: Dict[str, List[str]] = {}

    def clearcache(self):
        """Clear the cache entirely."""
        self.cache.clear()
        self.original_lines.clear()

    def _bisect_mapped_lineno(self, opcodes: List[Tuple[int, int, int, int]], lineno: int):
        if not opcodes:
            return -1
        line_idx = lineno - 1
        lo = 0
        hi = len(opcodes) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if (opcodes[mid][0] <= line_idx and line_idx < opcodes[mid][1]):
                # plus 1 because lineno is 1-based
                return opcodes[mid][2] + line_idx - opcodes[mid][0] + 1
            if opcodes[mid][0] < line_idx:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def query_mapped_linenos(self, filename, lineno: int, module_globals=None):
        """Get the lines for a Python source file from the cache.
        Update the cache if it doesn't contain an entry for this file already."""
        self.checkcache(filename)
        if filename in self.cache:
            entry = self.cache[filename]
            if entry.diff_opcodes_equal is None:
                # no diff info, return original lineno
                return lineno
            return self._bisect_mapped_lineno(entry.diff_opcodes_equal, lineno)
        try:
            success = self.updatecache(filename, module_globals)
            if not success:
                print("HERE?")
                return -1
            entry = self.cache[filename]
            if entry.diff_opcodes_equal is None:
                # no diff info, return original lineno
                return lineno
            return self._bisect_mapped_lineno(entry.diff_opcodes_equal, lineno)
        except MemoryError:
            self.clearcache()
            print("HERE?2")

            return -1

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
            if entry.mtime is None:
                continue   # no-op for files loaded via a __loader__
            try:
                stat = os.stat(entry.fullname)
            except OSError:
                self.cache.pop(filename, None)
                continue
            if entry.size != stat.st_size or entry.mtime != stat.st_mtime:
                self.cache.pop(filename, None)


    def updatecache(self, filename, module_globals=None):
        """Update a cache entry and return its list of lines.
        If something's wrong, print a message, discard the cache entry,
        and return an empty list."""

        if filename in self.cache:
            self.cache.pop(filename, None)
        if not filename or (filename.startswith('<') and filename.endswith('>')):
            return False

        fullname = filename
        try:
            stat = os.stat(fullname)
        except OSError:
            basename = filename
            # Try looking through the module search path, which is only useful
            # when handling a relative filename.
            if os.path.isabs(filename):
                return False

            for dirname in sys.path:
                try:
                    fullname = os.path.join(dirname, basename)
                except (TypeError, AttributeError):
                    # Not sufficiently string-like to do anything useful with.
                    continue
                try:
                    stat = os.stat(fullname)
                    break
                except OSError:
                    pass
            else:
                return False
        try:
            with tokenize.open(fullname) as fp:
                lines = fp.readlines()
        except (OSError, UnicodeDecodeError, SyntaxError):
            return False
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        eq_opcodes: Optional[List[Tuple[int, int, int, int]]] = None
        if filename not in self.original_lines:
            self.original_lines[filename] = lines
        else:
            eq_opcodes = []
            s = SequenceMatcher(self._is_junk, self.original_lines[filename], lines)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == 'equal':
                    eq_opcodes.append((i1, i2, j1, j2))

        size, mtime = stat.st_size, stat.st_mtime
        self.cache[filename] = SCDItem(size, mtime, lines, eq_opcodes, fullname)
        return True

    def _is_junk(self, x: str):
        # ignore empty lines and python comment lines
        x_strip = x.strip()
        return len(x_strip) == 0 or x_strip.startswith("#")
