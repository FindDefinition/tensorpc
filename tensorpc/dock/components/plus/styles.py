from tensorpc.dock.components import mui

class CodeStyles:
    fontFamily = "IBMPlexMono,SFMono-Regular,Consolas,Liberation Mono,Menlo,Courier,monospace"

def get_tight_icon_tab_theme(size: str = "28px"):
    tab_theme = mui.Theme(
        components={
            "MuiTab": {
                "styleOverrides": {
                    "root": {
                        "padding": "0",
                        "minWidth": size,
                        "minHeight": size,
                    }
                }
            }
        })
    return tab_theme


def get_tight_icon_tab_theme_horizontal(size: str = "28px", padding: str = "0px"):
    tab_theme = mui.Theme(
        components={
            "MuiTab": {
                "styleOverrides": {
                    "root": {
                        "padding": padding,
                        "minWidth": size,
                        "minHeight": size,
                    }
                }
            },
            "MuiTabs": {
                "styleOverrides": {
                    "root": {
                        "minHeight": size,
                    }
                }
            },
        })
    return tab_theme

def get_tight_tab_theme_horizontal(size: str = "28px", tab_padding: str = "5px"):
    tab_theme = mui.Theme(
        components={
            "MuiTab": {
                "styleOverrides": {
                    "root": {
                        "padding": f"0 {tab_padding} 0 {tab_padding}",
                        "minWidth": size,
                        "minHeight": size,
                    }
                }
            },
            "MuiTabs": {
                "styleOverrides": {
                    "root": {
                        "minHeight": size,
                    }
                }
            },
        })
    return tab_theme
