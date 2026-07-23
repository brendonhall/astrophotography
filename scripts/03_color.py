#!/usr/bin/env python3
"""Step 3 - Color calibration (still linear).

Two light-touch corrections appropriate for a first pass on OSC data:
  1. Background neutralization - force each channel's sky level to the same
     pedestal so the background is a neutral gray (kills the green cast).
  2. Gentle channel white-balance - equalize the mid-signal level across
     channels, but clamp the correction so we don't wreck star colors the
     SeeStar's own color matrix already got roughly right.

A full photometric calibration (matching star colors to a catalog via the WCS)
is the more advanced next step; noted for later.
"""
import sys
import numpy as np
import astrolib as al

PEDESTAL = 0.10 * 65535.0
GAIN_CLAMP = (0.85, 1.15)


def bg_level(chan):
    """Sigma-clipped median of the background (source pixels excluded)."""
    bg = chan[~al.source_mask(chan)]
    med = np.median(bg)
    for _ in range(3):
        s = bg.std()
        bg = bg[np.abs(bg - med) < 3 * s]
        med = np.median(bg)
    return med


def main(infile, outfile):
    img, hdr = al.load(infile)
    out = np.empty_like(img)

    # 1. neutralize background
    print("  background levels (ADU):", end=" ")
    for c, name in enumerate("RGB"):
        b = bg_level(img[..., c])
        print(f"{name}={b:.1f}", end="  ")
        out[..., c] = img[..., c] - b + PEDESTAL
    print()

    # 2. gentle WB: match mid-signal (p60..p99 of luminance) across channels to green
    lum = out.mean(axis=2)
    lo, hi = np.percentile(lum, 60), np.percentile(lum, 99)
    band = (lum > lo) & (lum < hi)
    means = np.array([out[..., c][band].mean() - PEDESTAL for c in range(3)])
    gains = means[1] / means  # normalize to green
    gains = np.clip(gains, *GAIN_CLAMP)
    print(f"  white-balance gains (R,G,B) = {gains.round(3)}")
    for c in range(3):
        out[..., c] = (out[..., c] - PEDESTAL) * gains[c] + PEDESTAL

    al.save(outfile, out, hdr)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
