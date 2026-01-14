import enum 
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from typing_extensions import Literal, TypeAlias, TypedDict, Self
from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterable,
                    Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, TypeVar, Union)
from aiortc import (
    MediaStreamTrack,
)

import dataclasses as dataclasses_plain

NumberType: TypeAlias = Union[int, float]

class IconType(enum.IntEnum):
    RestartAlt = 0
    Menu = 1
    Settings = 2
    Save = 3
    Close = 4
    ExpandMore = 5
    ExpandLess = 6
    Add = 7
    ChevronLeft = 8
    ChevronRight = 9
    Delete = 10
    AddCard = 11
    Clear = 12
    Fullscreen = 13
    PlayArrow = 14
    Pause = 15
    Stop = 16
    MoreVert = 17
    FullscreenExit = 18
    Code = 19
    Terminal = 20
    Videocam = 21
    CameraAlt = 22
    DragHandle = 23
    Dataset = 24
    DataObject = 25
    DataArray = 26
    Cached = 27
    SwapVert = 28
    Refresh = 29
    Grid3x3 = 30
    Help = 31
    Visibility = 32
    Mic = 33
    PlayCircleOutline = 34
    DragIndicator = 35
    Cancel = 36
    Done = 37
    Preview = 38
    Build = 39
    VisibilityOff = 40
    ManageAccounts = 41
    AccountCircle = 42
    BugReport = 43
    Dashboard = 44
    DashboardCustomize = 45
    Check = 46
    ContentCopy = 47
    ContentPaste = 48
    ContentCut = 49
    TableView = 50
    Image = 51
    Merge = 52
    DoubleArrow = 53
    AccountTree = 54
    Timeline = 55
    FiberManualRecord = 56
    NavigateNext = 57
    NavigateBefore = 58
    SkipNext = 59
    SkipPrevious = 60
    RadioButtonChecked = 61
    StopCircleOutlined = 62
    Block = 63
    Download = 64
    Upload = 65
    Link = 66
    LinkOff = 67
    Search = 68
    Info = 69
    QueryStats = 70
    BarChart = 71
    Adb = 72
    CloudDownload = 73
    CloudUpload = 74
    Insights = 75
    KeyboardDoubleArrowRight = 76
    KeyboardDoubleArrowLeft = 77
    KeyboardArrowRight = 78
    KeyboardArrowLeft = 79
    KeyboardArrowDown = 80
    KeyboardArrowUp = 81
    KeyboardDoubleArrowDown = 82
    KeyboardDoubleArrowUp = 83
    Shortcut = 84
    Input = 85
    Output = 86
    # non-material icons
    Reactflow = 200
    Markdown = 201

@dataclasses.dataclass
class MenuItem:
    id: str
    label: Union[Undefined, str] = undefined
    # null icon has same effect as undefined,
    # the only difference is we ignore undefined by default,
    # so you can't use undefined to remove icon.
    icon: Optional[Union[IconType, Undefined, str]] = undefined
    inset: Union[Undefined, bool] = undefined
    iconSize: Union[Undefined, Literal["inherit", "large", "medium",
                                       "small"]] = undefined
    iconFontSize: Union[Undefined, NumberType, str] = undefined
    divider: Union[Undefined, bool] = undefined
    autoFocus: Union[Undefined, bool] = undefined
    disableAutoFocusItem: Union[Undefined, bool] = undefined
    confirmMessage: Union[str, Undefined] = undefined
    confirmTitle: Union[str, Undefined] = undefined
    disabled: Union[Undefined, bool] = undefined
    userdata: Union[Undefined, Any] = undefined


ALL_KEY_CODES = {
    'Backquote', 'Backslash', 'Backspace', 'BracketLeft', 'BracketRight', 'Comma', 'Digit0', 'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7', 'Digit8', 'Digit9', 'Equal', 'IntlBackslash', 'IntlRo', 'IntlYen', 'KeyA', 'KeyB', 'KeyC', 'KeyD', 'KeyE', 'KeyF', 'KeyG', 'KeyH', 'KeyI', 'KeyJ', 'KeyK', 'KeyL', 'KeyM', 'KeyN', 'KeyO', 'KeyP', 'KeyQ', 'KeyR', 'KeyS', 'KeyT', 'KeyU', 'KeyV', 'KeyW', 'KeyX', 'KeyY', 'KeyZ', 'Minus', 'Period', 'Quote', 'Semicolon', 'Slash',
    'AltLeft', 'AltRight', 'CapsLock', 'ContextMenu', 'ControlLeft', 'ControlRight', 'Enter', 'MetaLeft', 'MetaRight', 'ShiftLeft', 'ShiftRight', 'Space', 'Tab',
    'Convert', 'KanaMode', 'Lang1', 'Lang2', 'Lang3', 'Lang4', 'Lang5', 'NonConvert',
    'Delete', 'End', 'Help', 'Home', 'Insert', 'PageDown', 'PageUp',
    'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowUp',
    'NumLock', 'Numpad0', 'Numpad1', 'Numpad2', 'Numpad3', 'Numpad4', 'Numpad5', 'Numpad6', 'Numpad7', 'Numpad8', 'Numpad9', 'NumpadAdd', 'NumpadBackspace', 'NumpadClear', 'NumpadClearEntry', 'NumpadComma', 'NumpadDecimal', 'NumpadDivide', 'NumpadEnter', 'NumpadEqual', 'NumpadHash', 'NumpadMemoryAdd', 'NumpadMemoryClear', 'NumpadMemoryRecall', 'NumpadMemoryStore', 'NumpadMemorySubtract', 'NumpadMultiply', 'NumpadParenLeft', 'NumpadParenRight', 'NumpadStar', 'NumpadSubtract',
    'Escape', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'Fn', 'FnLock', 'PrintScreen', 'ScrollLock', 'Pause',
    'BrowserBack', 'BrowserFavorites', 'BrowserForward', 'BrowserHome', 'BrowserRefresh', 'BrowserSearch', 'BrowserStop', 'Eject', 'LaunchApp1', 'LaunchApp2', 'LaunchMail', 'MediaPlayPause', 'MediaSelect', 'MediaStop', 'MediaTrackNext', 'MediaTrackPrevious', 'Power', 'Sleep', 'AudioVolumeDown', 'AudioVolumeMute', 'AudioVolumeUp', 'WakeUp'
}


@dataclasses_plain.dataclass
class RTCTrackInfo:
    track: MediaStreamTrack
    kind: Literal["audio", "video"]
    force_codec: Optional[str] = None