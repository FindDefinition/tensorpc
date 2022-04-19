import os
import platform
import re
import subprocess
import sys
import sysconfig
from enum import Enum
from typing import Tuple

Python3 = (sys.version_info[0] == 3)
Python4 = (sys.version_info[0] == 4)
Python3AndLater = (sys.version_info[0] >= 3)
Python3Later = (sys.version_info[0] > 3)
Python35 = Python3 and sys.version_info[1] == 5
Python3_11AndLater = Python3Later or (Python3 and sys.version_info[1] >= 11)
Python3_10AndLater = Python3Later or (Python3 and sys.version_info[1] >= 10)
Python3_9AndLater = Python3Later or (Python3 and sys.version_info[1] >= 9)
Python3_8AndLater = Python3Later or (Python3 and sys.version_info[1] >= 8)
Python3_7AndLater = Python3Later or (Python3 and sys.version_info[1] >= 7)
Python3_6AndLater = Python3Later or (Python3 and sys.version_info[1] >= 6)
Python3_5AndLater = Python3Later or (Python3 and sys.version_info[1] >= 5)
PyPy3 = platform.python_implementation().lower() == "pypy"
assert Python3_5AndLater, "only support python >= 3.5"

VALID_PYTHON_MODULE_NAME_PATTERN = re.compile(r"[a-zA-Z_][0-9a-zA-Z_]*")


class OSType(Enum):
    Win10 = "Win10"
    MacOS = "MacOS"
    Linux = "Linux"
    Unknown = "Unknown"


OS = OSType.Unknown

InWindows = False
if os.name == 'nt':
    InWindows = True
    OS = OSType.Win10

InLinux = False
if platform.system() == "Linux":
    InLinux = True
    OS = OSType.Linux

InMacOS = False
if platform.system() == "Darwin":
    InMacOS = True
    OS = OSType.MacOS

