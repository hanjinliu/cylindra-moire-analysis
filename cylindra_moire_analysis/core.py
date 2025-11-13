from cylindra.plugin import register_function
from cylindra.widgets import CylindraMainWidget
from cylindra.annotations import SplineType, BinSizeType
from magicclass.ext.pyqtgraph import QtImageCanvas
import numpy as np

@register_function
def straighten_and_project(
    ui: CylindraMainWidget,
    spline: SplineType,
    bin_size: BinSizeType = 1,
):
    img_st = ui.tomogram.straighten(spline, binsize=bin_size)
    img_proj = img_st.mean(axis=0)
    ps_proj = img_proj.power_spectra(shape=(1024, 1024))
    logps_proj = np.log(ps_proj + 1e-4)
    canvas = QtImageCanvas()
    canvas.image = logps_proj
    canvas.show()
