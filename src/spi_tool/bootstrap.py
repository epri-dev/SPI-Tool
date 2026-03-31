from __future__ import annotations

import holoviews as hv
import matplotlib as mpl
import pandas as pd
import panel as pn


_BOOTSTRAPPED = False


def configure() -> None:
    global _BOOTSTRAPPED

    if _BOOTSTRAPPED:
        return

    pd.options.mode.copy_on_write = True
    pd.options.future.no_silent_downcasting = True

    mpl.use("agg")
    mpl.rcParams["figure.constrained_layout.use"] = True

    pn.extension(
        "tabulator",
        "perspective",
        "mathjax",
        "modal",
        sizing_mode="stretch_width",
        notifications=True,
        throttled=True,
    )
    pn.config.layout_compatibility = "error"
    hv.extension("bokeh")

    pn.pane.Markdown.styles = {"font-size": "16px", "line-height": "1.6"}
    pn.widgets.Button.styles = {"font-size": "16px"}

    _BOOTSTRAPPED = True
