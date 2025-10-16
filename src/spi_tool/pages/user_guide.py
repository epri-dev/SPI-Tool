import panel as pn
import os
import subprocess
import shlex
import shutil

from .. import _utils


manual_name = "SPI-Tool-Software-Manual.pdf"
manual_path = os.path.join(_utils.get_resources_path(), manual_name)


def _try_ensure_manual_present():
    try:
        if os.path.exists(manual_path):
            return

        os.makedirs(_utils.get_resources_path(), exist_ok=True)

        blobspec = "assets:assets/" + manual_name
        content = subprocess.check_output(
            ["git", "show", blobspec],
            stderr=subprocess.STDOUT,
        )

        with open(manual_path, "wb") as f:
            f.write(content)

    except Exception:
        pass


_try_ensure_manual_present()


class UserGuide(pn.viewable.Viewer):
    def __panel__(self):
        if os.path.exists(manual_path):
            return pn.pane.PDF(manual_path, width=700, height=1000)
        else:
            return pn.pane.Markdown(
                "⚠️ **User Guide not available.**\n\n"
                "The file `SPI-Tool-Software-Manual.pdf` could not be found"
            )
