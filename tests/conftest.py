

from pathlib import Path
import miaaim
from miaaim.io.imread._import import HDIreader

_data_small = "/Users/joshuahess/Desktop/miaaim-python-dev/tests/DFU_imc.ome.tiff"


@pytest.fixture
def imc_imported():
    """test import function"""

    d = HDIreader(path_to_data=_data_small.copy(),
            path_to_markers=None,
            flatten=False,
            subsample=None,
            mask=None,
            save_mem=False,
            method="random",
            n=10000,
            grid_spacing=(2, 2))
    assert isinstance(m, HDIreader), m

    return d
