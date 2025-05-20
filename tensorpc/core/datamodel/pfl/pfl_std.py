import dataclasses 
import math
import random
import struct
from typing import Union
from .pfl_reg import register_df_std

# implement all math func in javascript Math 

@register_df_std(js_name="Math")
@dataclasses.dataclass
class Math:
    @staticmethod 
    def abs(x: float) -> float:
        return abs(x)

    @staticmethod 
    def acos(x: float) -> float:
        return math.acos(x)

    @staticmethod
    def asin(x: float) -> float:
        return math.asin(x)

    @staticmethod
    def atan(x: float) -> float:
        return math.atan(x)

    @staticmethod
    def atan2(y: float, x: float) -> float:
        return math.atan2(y, x)

    @staticmethod
    def ceil(x: float) -> float:
        return math.ceil(x)

    @staticmethod
    def cos(x: float) -> float:
        return math.cos(x)

    @staticmethod
    def exp(x: float) -> float:
        return math.exp(x)

    @staticmethod
    def floor(x: float) -> float:
        return math.floor(x)

    @staticmethod
    def log(x: float) -> float:
        return math.log(x)

    @staticmethod
    def max(*args: float) -> float:
        return max(args)

    @staticmethod
    def min(*args: float) -> float:
        return min(args)

    @staticmethod
    def pow(x: float, y: float) -> float:
        return math.pow(x, y)

    @staticmethod
    def random() -> float:
        return random.random()

    @staticmethod
    def round(x: float) -> float:
        return round(x)

    @staticmethod
    def sign(x: float) -> int:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    @staticmethod
    def sin(x: float) -> float:
        return math.sin(x)

    @staticmethod
    def sqrt(x: float) -> float:
        return math.sqrt(x)

    @staticmethod
    def tan(x: float) -> float:
        return math.tan(x)

    @staticmethod
    def trunc(x: float) -> float:
        return math.trunc(x)

    @staticmethod
    def cbrt(x: float) -> float:
        return math.copysign(abs(x) ** (1/3), x)

    @staticmethod
    def clz32(x: int) -> int:
        return 32 - len(bin(x & 0xffffffff)[2:]) if x != 0 else 32

    @staticmethod
    def imul(a: int, b: int) -> int:
        return (a * b) & 0xffffffff if (a * b) >= 0 else ((a * b) + 0x100000000) & 0xffffffff

    @staticmethod
    def fround(x: float) -> float:
        return struct.unpack('f', struct.pack('f', x))[0]

    @staticmethod
    def log10(x: float) -> float:
        return math.log10(x)

    @staticmethod
    def log2(x: float) -> float:
        return math.log2(x)

    @staticmethod
    def log1p(x: float) -> float:
        return math.log1p(x)

    @staticmethod
    def expm1(x: float) -> float:
        return math.expm1(x)

    @staticmethod
    def hypot(*args: float) -> float:
        return math.hypot(*args)

    @staticmethod
    def sinh(x: float) -> float:
        return math.sinh(x)

    @staticmethod
    def cosh(x: float) -> float:
        return math.cosh(x)

    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def asinh(x: float) -> float:
        return math.asinh(x)

    @staticmethod
    def acosh(x: float) -> float:
        return math.acosh(x)

    @staticmethod
    def atanh(x: float) -> float:
        return math.atanh(x)

    @staticmethod
    def to_degrees(x: float) -> float:
        return math.degrees(x)

    @staticmethod
    def to_radians(x: float) -> float:
        return math.radians(x)

    # Constants
    E: float = math.e
    LN10: float = math.log(10)
    LN2: float = math.log(2)
    LOG2E: float = math.log2(math.e)
    LOG10E: float = math.log10(math.e)
    PI: float = math.pi
    SQRT1_2: float = math.sqrt(0.5)
    SQRT2: float = math.sqrt(2)


@register_df_std(js_name="MathUtil")
@dataclasses.dataclass
class MathUtil:
    @staticmethod
    def clamp(x: float, min_val: float, max_val: float) -> float:
        return max(min(x, max_val), min_val)