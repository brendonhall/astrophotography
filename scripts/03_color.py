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
import numpy as np
import astrolib as al
import pcc

PEDESTAL = 0.10 * 65535.0
REF_BP_RP = 0.82
MIN_STARS = 30


def bg_level(chan):
    """Sigma-clipped median of the background (source pixels excluded)."""
    bg = chan[~al.source_mask(chan)]
    med = np.median(bg)
    for _ in range(3):
        s = bg.std()
        bg = bg[np.abs(bg - med) < 3 * s]
        med = np.median(bg)
    return med


def gentle_white_balance(img, pedestal, clamp=(0.85, 1.15)):
    """Match mid-signal (p60..p99 luminance) across channels to green; clamped."""
    lum = img.mean(axis=2)
    lo, hi = np.percentile(lum, 60), np.percentile(lum, 99)
    band = (lum > lo) & (lum < hi)
    means = np.array([img[..., c][band].mean() - pedestal for c in range(3)])
    gains = np.clip(means[1] / means, *clamp)
    out = img.copy()
    for c in range(3):
        out[..., c] = (out[..., c] - pedestal) * gains[c] + pedestal
    print(f"  gentle-WB gains (R,G,B) = {gains.round(3)}")
    return out


def main(infile, outfile, original=None, no_pcc=False, diagnostic=None):
    img, hdr = al.load(infile)
    out = np.empty_like(img)

    # 1. neutralize background
    print("  background levels (ADU):", end=" ")
    for c, name in enumerate("RGB"):
        b = bg_level(img[..., c])
        print(f"{name}={b:.1f}", end="  ")
        out[..., c] = img[..., c] - b + PEDESTAL
    print()

    # Keep a pristine, never-mutated copy of the neutralized image so any
    # fallback (PCCError or an unexpected failure after gains were computed,
    # e.g. in save_diagnostic) always starts from the same starting point --
    # never from an already color-corrected array (which would silently
    # double-apply a correction).
    neutralized = out.copy()

    # 2. color balance: PCC from the original (WCS-bearing) stack, else gentle WB
    diag_path = diagnostic or outfile.replace(".fit", "_pcc_diagnostic.png")
    if original and not no_pcc:
        try:
            oimg, ohdr = al.load(original)
            gains, report = pcc.photometric_calibration(
                oimg, ohdr, ref_bp_rp=REF_BP_RP, min_stars=MIN_STARS)
            # Write the diagnostic BEFORE applying gains: if this raises, we
            # fall through to the fallback below instead of having already
            # mutated `out` with a correction we then can't report cleanly.
            pcc.save_diagnostic(report, diag_path)
            result = pcc.apply_gains(neutralized, gains, PEDESTAL)
            out = result
            print(f"  PCC gains (R,G,B) = ({gains[0]:.3f}, 1.000, {gains[2]:.3f}) "
                  f"from {report['n_matched']} stars")
        except pcc.PCCError as e:
            print(f"  WARNING: PCC unavailable ({e}); using gentle white balance")
            out = gentle_white_balance(neutralized, PEDESTAL)
        except Exception as e:
            print(f"  WARNING: PCC attempt failed ({type(e).__name__}: {e}); using gentle white balance")
            out = gentle_white_balance(neutralized, PEDESTAL)
    else:
        out = gentle_white_balance(neutralized, PEDESTAL)

    al.save(outfile, out, hdr)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--original", help="original stacked FITS (with WCS) for PCC")
    ap.add_argument("--no-pcc", action="store_true")
    ap.add_argument("--diagnostic", help="path for the PCC color-color diagnostic PNG "
                                          "(default: <outfile>_pcc_diagnostic.png)")
    a = ap.parse_args()
    main(a.infile, a.outfile, original=a.original, no_pcc=a.no_pcc, diagnostic=a.diagnostic)
