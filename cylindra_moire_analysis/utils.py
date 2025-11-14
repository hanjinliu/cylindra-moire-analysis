from __future__ import annotations
from typing import NamedTuple
from scipy.signal import find_peaks
from scipy.fft import next_fast_len
import numpy as np
import impy as ip

class Peak(NamedTuple):
    index: int
    yvalue: float
    is_maximum: bool
    
def _find_peaks(data: np.ndarray, thresh_rel: float = 0.33) -> list[Peak]:
    prof_sq = (data - data.mean()) ** 2
    ddy = np.gradient(np.gradient(data))
    i_peaks, _ = find_peaks(prof_sq)
    peaks = [
        Peak(_i, data[_i], ddy[_i] < 0) for _i in i_peaks
        if prof_sq[_i] > thresh_rel * prof_sq.max()
    ]
    return peaks

def find_min_near_center(data: np.ndarray) -> int:
    peaks = _find_peaks(data, thresh_rel=0.01)
    peaks = [p for p in peaks if not p.is_maximum]
    if len(peaks) == 0:
        return np.argmin(data[2:-2]) + 2
    i_center = (data.size - 1) / 2
    dist_from_center = [abs(p.index - i_center) for p in peaks]    
    return peaks[np.argmin(dist_from_center)].index

def filter_filament(img_proj: ip.ImgArray):
    fft_size = next_fast_len(img_proj.shape.y)
    ymax = int(img_proj.scale.y * fft_size / 100)
    
    ft_shape = (fft_size, fft_size)
    img_ft = img_proj.fft(shift=False, shape=ft_shape)
    
    xmin, xmax = 1, int(fft_size * 0.4)
    slices_to_copy = [
        (slice(0, ymax), slice(xmin, xmax)),
        (slice(-ymax + 1, None), slice(xmin, xmax)),
    ]
    img_ft_masked = np.zeros_like(img_ft)
    for sl in slices_to_copy:
        img_ft_masked[sl] = img_ft[sl]
    img_filt = img_ft_masked.ifft(shift=False)[: img_proj.shape[0], : img_proj.shape[1]]
    return img_filt, img_ft
