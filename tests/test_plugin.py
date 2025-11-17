from cylindra.widgets import CylindraMainWidget
from cylindra_builtins.fetch import mt_14pf

from cylindra_moire_analysis import (
    export_for_tubulej,
    measure_skew,
    save_results_as_csv,
)


def test_functions(ui: CylindraMainWidget, tmpdir):
    mt_14pf(ui, with_spline="fitted")
    measure_skew(ui, [0])
    export_for_tubulej(ui, tmpdir, [0])
    save_results_as_csv(ui, tmpdir / "results.csv")
