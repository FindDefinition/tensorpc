import asyncio
import dataclasses 


from tensorpc.flow import mui, mark_create_layout
from tensorpc.flow.components.plus.styles import (CodeStyles,
                                                  get_tight_icon_tab_theme)


@dataclasses.dataclass
class ListWithPeriodScan:
    data_list: mui.DataFlexBox
    lock: asyncio.Lock
    