#!/usr/bin/env python3
"""Step 2 - Remove the light-pollution gradient (background extraction).

Model each channel's background as a low-order 2D polynomial fit ONLY to
non-source pixels (stars/galaxy masked out), then subtract it and add back a
common neutral pedestal. This is the Python analogue of Siril's polynomial
'Background Extraction' / 'subsky'.

Why a polynomial: real LP gradients are smooth and low-frequency, so a degree-2
or -3 surface captures them without touching the galaxy's actual structure.
"""
import sys
import numpy as np
import astrolib as al

DEGREE = 3          # total-degree of the 2D polynomial
SAMPLE = 12         # fit on every Nth background pixel (speed)
PEDESTAL = 0.10     # re-added background level, as fraction of 65535 -> keeps data positive


def poly_design(x, y, degree):
    """Columns for all terms x^i y^j with i+j <= degree."""
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.stack(cols, axis=-1)


def fit_background(chan, degree, sample):
    h, w = chan.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # normalize coords to [-1,1] for numerical stability
    xn = (xx / (w - 1)) * 2 - 1
    yn = (yy / (h - 1)) * 2 - 1

    mask = al.source_mask(chan)
    bg = ~mask
    xs, ys, vs = xn[bg][::sample], yn[bg][::sample], chan[bg][::sample]
    A = poly_design(xs, ys, degree)
    coef, *_ = np.linalg.lstsq(A, vs, rcond=None)

    full = poly_design(xn.ravel(), yn.ravel(), degree) @ coef
    return full.reshape(h, w), mask.mean()


def main(infile, outfile):
    img, hdr = al.load(infile)
    pedestal = PEDESTAL * 65535.0
    out = np.empty_like(img)
    print(f"  {'chan':5s} {'bg slope range (ADU)':>22s} {'src%':>6s}")
    for c, name in enumerate("RGB"):
        model, srcfrac = fit_background(img[..., c], DEGREE, SAMPLE)
        rng = model.max() - model.min()
        out[..., c] = img[..., c] - model + pedestal
        print(f"  {name:5s} {rng:22.1f} {srcfrac*100:6.2f}")
    al.save(outfile, out, hdr)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
