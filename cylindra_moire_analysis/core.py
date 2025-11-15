from typing import Annotated

import impy as ip
import numpy as np
from cylindra.annotations import BinSizeType, SplinesType
from cylindra.plugin import register_function
from cylindra.widgets import CylindraMainWidget
from magicclass.types import Path
from matplotlib import pyplot as plt

from cylindra_moire_analysis.utils import filter_filament, find_min_near_center


@register_function(name="Measure skew ...")
def measure_skew(
    ui: CylindraMainWidget,
    splines: SplinesType,
    bin_size: BinSizeType = 1,
    filament_width: Annotated[float, {"label": "filament width (nm)"}] = 30.0,
    dx: Annotated[float, {"label": "inter-protofilament distance (nm)", "min": 1, "max": 100, "step": 0.001}] = 4.895,
    min_moier_periodicity: Annotated[float, {"label": "minimum moire periodicity (nm)"}] = 200.0,
):  # fmt: skip
    """Measure skew angle in the traditional way that uses moire pattern.

    This method does essentially the same analysis as described in Ku et al., 2020,
    Bio-protocols, with some autmations.

    Parameters
    ----------
    splines : SplinesType
        Spline indices to be analyzed.
    bin_size : int, default 1
        Binning factor.
    filament_width : nm, default 30.0
        Width of the straightened images. Must be larger than the diameter of the
        filament but small enough to exclude noise.
    dx : nm, default 4.895
        Inter-protofilament distance.
    min_moier_periodicity : nm, default 200.0
        Minimum moire periodicity to consider. Setting to larger value will restrict
        the search range and may help to avoid false detection.
    """
    img_st_list = ui.tomogram.straighten(splines, binsize=bin_size, size=filament_width)
    props: dict[int, dict] = {}
    ui.logger.print_html("<h3> --- Moire pattern analysis --- </h3>")
    for i, img_st in zip(splines, img_st_list, strict=False):
        spl = ui.splines[i]
        skew_sign = np.sign(spl.props.get_glob("skew_angle", 1))
        ui.logger.print_html(f"<b>Spline-{i}</b>")
        ui.logger.print(f"Length = {spl.length():.1f} nm")
        img_proj = img_st.mean(axis=0)
        img_filt, img_ft = filter_filament(img_proj)
        ps_sl = slice(0, int(0.53 * img_ft.shape[0]))
        ps_proj = np.fft.fftshift(img_ft.real**2 + img_ft.imag**2)[ps_sl]

        with ui.logger.set_plt():
            plt.figure(figsize=(4, 4))
            plt.title("Power spectrum")
            plt.imshow(
                ps_proj, cmap="inferno", vmin=0, vmax=np.percentile(ps_proj, 99.9)
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        with ui.logger.set_plt():
            _, axes = plt.subplots(nrows=2, figsize=(6, 3.2), sharex=True)

            # find the horizontal line that has the maximum amplitude in the real space
            plt.sca(axes[0])
            iyopt = np.argmax(np.std(img_filt, axis="x"))
            ix = find_min_near_center(img_filt[iyopt])
            profile = img_filt[f"x={ix}"]
            plt.imshow(img_filt.T, cmap="gray")
            plt.axhline(ix + 0.5, color="lime", lw=1)

            plt.sca(axes[1])
            plt.plot(profile, color="green")
            plt.xlabel("Position (px)")
            plt.tight_layout(h_pad=0.0)
            plt.show()

            # power spectrum analysis
            plt.figure(figsize=(5, 2))
            ymax_loc_ps = int(
                round(img_proj.shape.y * img_proj.scale.y / min_moier_periodicity)
            )
            upsample_factor = 50
            spec = (profile - profile.mean()).local_power_spectra(
                f"y=0:{ymax_loc_ps}", upsample_factor=upsample_factor
            )
            wnum = np.argmax(spec) / upsample_factor
            plt.title("Power spectrum of the line profile")
            plt.plot(np.arange(spec.size) / upsample_factor, spec, color="gray")
            plt.axvline(wnum, color="red", lw=1, ls="--")
            plt.show()

        if wnum == 0:
            skew_moire = 0.0
        else:
            dist = 1 / wnum * img_proj.scale.y * img_proj.shape.y
            ui.logger.print(f"L = {dist:.1f} nm, δx = {dx:.3f} nm")
            skew_moire = np.rad2deg(np.arcsin(min(dx / dist, 1)))
        props[i] = {"moire_skew_angle": skew_moire * skew_sign}

        for prop in props.values():
            spl.props.update_glob(prop, bin_size=bin_size)

    table = [[""] + list(next(iter(props.values())).keys())]
    for i, prop in props.items():
        row = [f"spline-{i}"] + [f"{v:.3f}" for v in prop.values()]
        table.append(row)
    ui.logger.print_html("<b><u>Summary</u></b>")
    ui.logger.print_table(table, header=False)


@register_function(name="Export for TubuleJ ...", record=False)
def export_for_tubulej(
    ui: CylindraMainWidget,
    save_dir: Path.Dir,
    splines: SplinesType,
    bin_size: BinSizeType = 1,
    filament_width: Annotated[float, {"label": "filament width (nm)"}] = 30.0,
    save_filtered_images: bool = True,
    project_prefix: str = "MT_",
):  # fmt: skip
    """Export the straightened and filtered images.

    This method creates directories in a format compatible with the manual analysis
    using TubuleJ.

    Parameters
    ----------
    save_dir : Path.Dir
        The root directory to save microtubule project directories.
    splines : SplinesType
        Spline indices to be analyzed.
    bin_size : int, default 1
        Binning factor.
    filament_width : nm, default 30.0
        Width of the straightened images. Must be larger than the diameter of the
        filament but small enough to exclude noise.
    save_filtered_images : bool, default True
        Also save the moire-filtered images.
    project_prefix : str, default "MT_"
        Prefix for the microtubule project directories.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Directory '{save_dir}' does not exist.")
    img_st_list = ui.tomogram.straighten(splines, binsize=bin_size, size=filament_width)
    for i, img_st in zip(splines, img_st_list, strict=False):
        mt_dir = save_dir / f"{project_prefix}{i}"
        mt_dir.mkdir(exist_ok=True)
        img_proj = img_st.mean(axis=0)
        _rot90(img_proj).imsave(mt_dir / f"{i}-straight_Centered.tif")
        if save_filtered_images:
            img_filt, _ = filter_filament(img_proj)
            _rot90(img_filt).imsave(mt_dir / "filtered image.tif")
        angst = img_st.scale.y * 10
        mt_dir.joinpath("calibration.csv").write_text(_calib_text(angst))


def _rot90(img: ip.ImgArray) -> ip.ImgArray:
    """Rotate image by 90 degrees counter-clockwise."""
    return np.rot90(img)


def _calib_text(angst: float) -> str:
    """Generate calibration text for TubuleJ."""
    # NOTE: TubuleJ expects calibration file is a csv file with tab delimiter!
    return f"pixelSize\tunit\n{angst:.2f}\t?\n"
