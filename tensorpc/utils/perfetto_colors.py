
import abc
import dataclasses
import math
import re
from typing import Union, Optional, Tuple
from typing_extensions import TypeAlias, Self
import hsluv
import numpy as np 

LIGHTNESS_MIN = 0
LIGHTNESS_MAX = 100

SATURATION_MIN = 0
SATURATION_MAX = 100


def _hash(s: str, max: int) -> int:
    hash = 0x811c9dc5 & 0xfffffff
    for i in range(len(s)):
        # print("HASH-0", i, hash, ord(s[i]), )
        hash ^= ord(s[i])
        # print("HASH-1",hash)
        hash = int(np.array(int(float(hash) * float(16777619)) & 0xffffffff, dtype=np.uint64).astype(np.int32))
        # hash = to_js_safe_int(hash * 16777619) & 0xffffffff
    return int(abs(hash) % max)


NumberType: TypeAlias = Union[int, float]

@dataclasses.dataclass
class HSL:
    h: NumberType
    s: NumberType
    l: NumberType

def hsl_to_rgb(h: NumberType, s: NumberType, l: NumberType) -> Tuple[NumberType, NumberType, NumberType]:
    s = s / SATURATION_MAX
    l = l / LIGHTNESS_MAX
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((math.fmod((h / 60), 2)) - 1))
    m = l - c / 2
    r, g, b = 0, 0, 0
    if 0 <= h and h < 60:
        r, g, b = c, x, 0
    elif 60 <= h and h < 120:
        r, g, b = x, c, 0
    elif 120 <= h and h < 180:
        r, g, b = 0, c, x
    elif 180 <= h and h < 240:
        r, g, b = 0, x, c
    elif 240 <= h and h < 300:
        r, g, b = x, 0, c
    elif 300 <= h and h < 360:
        r, g, b = c, 0, x
    r = round((r + m) * 255)
    g = round((g + m) * 255)
    b = round((b + m) * 255)
    return r, g, b
    
def hex_to_rgb(hex: str) -> Tuple[int, int, int]:
    if len(hex) == 4:
        return int(hex[1] * 2, 16), int(hex[2] * 2, 16), int(hex[3] * 2, 16)
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)

def rgb_to_hsl(r: NumberType, g: NumberType, b: NumberType) -> Tuple[NumberType, NumberType, NumberType]:
    r = r / 255
    g = g / 255
    b = b / 255
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    h = (max_val + min_val) / 2
    s = (max_val + min_val) / 2
    l = (max_val + min_val) / 2
    if max_val == min_val:
        h = 0
        s = 0
    else:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        elif max_val == b:
            h = (r - g) / d + 4
        h = h / 6
    h = (h * 360)
    s = (s * 100)
    l = (l * 100)
    return h, s, l

@dataclasses.dataclass
class Color(abc.ABC):
    cssString: str
    perceivedBrightness: NumberType
    rgb: tuple[NumberType, NumberType, NumberType]

    @abc.abstractmethod
    def lighten(self, amount: NumberType, max_val: NumberType = LIGHTNESS_MAX) -> Self:
        ...

    @abc.abstractmethod
    def darken(self, amount: NumberType, min_val: NumberType = LIGHTNESS_MIN) -> Self:
        ... 

    @abc.abstractmethod
    def saturate(self, amount: NumberType, max_val: NumberType = LIGHTNESS_MAX) -> Self:
        ...

    @abc.abstractmethod
    def desaturate(self, amount: NumberType, min_val: NumberType = LIGHTNESS_MIN) -> Self:
        ... 


def clamp(val: NumberType, min_val: NumberType, max_val: NumberType) -> NumberType:
    return max(min(val, max_val), min_val)

class HSLColorBase(Color):

    def __init__(self, init: Union[tuple[NumberType, ...], HSL, str], alpha: Optional[NumberType]):
        if isinstance(init, (list, tuple)):
            self.hsl = init 
        elif isinstance(init, str):
            rgb = hex_to_rgb(init)
            self.hsl = rgb_to_hsl(*rgb)
        else:
            self.hsl = (init.h, init.s, init.l)
        self.alpha = alpha

    def lighten(self, amount: NumberType, max_val: NumberType = LIGHTNESS_MAX) -> Self:
        newLightness = clamp(self.hsl[2] + amount, LIGHTNESS_MIN, max_val)
        return self.__class__((self.hsl[0], self.hsl[1], newLightness, ), self.alpha)

    def darken(self, amount: NumberType, min_val: NumberType = LIGHTNESS_MIN) -> Self:
        newLightness = clamp(self.hsl[2] - amount, min_val, LIGHTNESS_MAX)
        return self.__class__((self.hsl[0], self.hsl[1], newLightness, ), self.alpha)

    def saturate(self, amount: NumberType, max_val: NumberType = SATURATION_MAX) -> Self:
        newSaturation = clamp(self.hsl[1] + amount, SATURATION_MIN, max_val)
        return self.__class__((self.hsl[0], newSaturation, self.hsl[2]), self.alpha)

    def desaturate(self, amount: NumberType, min_val: NumberType = SATURATION_MIN) -> Self:
        newSaturation = clamp(self.hsl[1] - amount, min_val, SATURATION_MAX)
        return self.__class__((self.hsl[0], newSaturation, self.hsl[2]), self.alpha)

def perceivedBrightness(r: NumberType, g: NumberType, b: NumberType) -> NumberType:
  # YIQ calculation from https://24ways.org/2010/calculating-color-contrast
  return (r * 299 + g * 587 + b * 114) / 1000


class HSLuvColor(HSLColorBase):
    def __init__(self, hsl: Union[tuple[NumberType, ...], HSL], alpha: Optional[NumberType] = None):
        super().__init__(hsl, alpha)
        rgb = hsluv.hsluv_to_rgb(self.hsl)
        r = math.floor(rgb[0] * 255)
        g = math.floor(rgb[1] * 255)
        b = math.floor(rgb[2] * 255)
        self.perceivedBrightness = perceivedBrightness(r, g, b)
        if self.alpha is None:
            self.cssString = f"rgb({r}, {g}, {b})"
        else:
            self.cssString = f"rgba({r}, {g}, {b} / {self.alpha})"
        self.rgb = (r, g, b)

class HSLColor(HSLColorBase):
    def __init__(self, hsl: Union[tuple[NumberType, ...], HSL], alpha: Optional[NumberType] = None):
        super().__init__(hsl, alpha)
        rgb = hsl_to_rgb(*self.hsl)
        r = math.floor(rgb[0] * 255)
        g = math.floor(rgb[1] * 255)
        b = math.floor(rgb[2] * 255)
        self.perceivedBrightness = perceivedBrightness(r, g, b)
        if self.alpha is None:
            self.cssString = f"rgb({r}, {g}, {b})"
        else:
            self.cssString = f"rgba({r}, {g}, {b} / {self.alpha})"
        self.rgb = (r, g, b)

WHITE_COLOR = HSLColor((0, 0, 100))
BLACK_COLOR = HSLColor((0, 0, 0))
GRAY_COLOR = HSLColor((0, 0, 90))

@dataclasses.dataclass
class ColorScheme:
    base: Color
    variant: Color
    disabled: Color
    textBase: Color
    textVariant: Color
    textDisabled: Color

PROCEDURAL_COLOR_CACHE: dict[str, ColorScheme] = {}
PERCEIVED_BRIGHTNESS_LIMIT = 180

def make_color_scheme(base: Color, variant: Optional[Color] = None):
    if variant is None:
        variant = base.darken(15).saturate(15)
    return ColorScheme(
        base,
        variant,
        GRAY_COLOR,
        BLACK_COLOR if base.perceivedBrightness >= PERCEIVED_BRIGHTNESS_LIMIT else WHITE_COLOR,
        BLACK_COLOR if variant.perceivedBrightness >= PERCEIVED_BRIGHTNESS_LIMIT else WHITE_COLOR,
        WHITE_COLOR,
    )

_REGEX = re.compile(r"( )?\d+")

def create_slice_name(name: str):
    return _REGEX.sub("", name).strip()

def perfetto_string_to_color(seed: str, use_cache: bool = False):
    if use_cache and seed in PROCEDURAL_COLOR_CACHE:
        return PROCEDURAL_COLOR_CACHE[seed]
    h = _hash(seed, 360)
    saturation = 80
    # print("HUE", h, seed)
    base = HSLuvColor((h, saturation, _hash(seed + "x", 40) + 40))
    variant = HSLuvColor((h, saturation, 30))
    colorScheme = make_color_scheme(base, variant)
    if use_cache:
        PROCEDURAL_COLOR_CACHE[seed] = colorScheme
    return colorScheme

def perfetto_slice_to_color(seed: str, use_cache: bool = False):
    seed = create_slice_name(seed)
    return perfetto_string_to_color(seed, use_cache)

def _main():
    print(perfetto_string_to_color("data").base)
    # print(perfetto_slice_to_color("fwdbwd"))
    # print(perfetto_slice_to_color("optim"))
    a = 1628185618
    b = 16777619
    a_double = np.double(a)
    b_double = np.double(b)
    mul_res = int(a_double * b_double)
    mul_res_u64 = np.int64(mul_res)
    print(mul_res_u64 & np.int64(0xffffffff))

if __name__ == "__main__":
    _main()