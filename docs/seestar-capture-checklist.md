# SeeStar S30 Pro — Capture Prep Checklist

Pre-session checklist for capturing deep-sky targets with the goal of feeding
individual subframes into the re-stacking + Python processing pipeline. Verify
exact app wording against your current firmware — UI moves between versions.

## Before leaving / at setup
- [ ] Battery charged; bring a power bank for long sessions.
- [ ] Enough **storage free** for all-frames: hundreds of ~50 MB FITS = several GB.
      Check the SeeStar's storage (and microSD if used) before starting.
- [ ] Tripod **leveled** on stable ground.
- [ ] Let the scope **thermally settle** to ambient (~10–15 min) before darks — the
      sensor is uncooled, so dark current tracks temperature.

## One-time / new-unit calibration
- [ ] Run the app's **compass / level (gyro) calibration** (figure-8 motion) for
      good GoTo and framing.
- [ ] Let it **autofocus** at the start of the session.

## Image calibration — DARK FRAMES (the one that matters)
- [ ] Shoot a fresh **dark library** with the app's built-in dark routine (cap on),
      at tonight's temperature. The SeeStar auto-applies darks to the stack and subs.
- [ ] Re-shoot darks if a later night is much warmer/colder than this library.
- [ ] No flats needed (sealed optics; gradient removed in post by pipeline step 02).
- [ ] No separate bias frames (folded into darks).

## Critical settings for the re-stack workflow
- [ ] **Enter Stargazing (deep-sky) mode**, pick the target, go to the imaging screen.
- [ ] On that screen, open the **gear/settings**, and turn ON **"Save all frames"**
      (saves each sub as FITS). **Must be enabled BEFORE starting** — subs can't be
      recovered afterward. You still get the live stack too.
- [ ] Enable **dithering** if offered — with darks, it kills fixed-pattern / walking
      noise and helps rejection scrub satellite trails.

## Capture plan
- [ ] Target: **M101** (Pinwheel). Aim for **longer total integration** than the
      first attempt (was 283 × 30s ≈ 2.4 h). More time = lower noise = harder stretch
      = fainter arm/tidal detail. 5–6 h+ across nights is a big step up.
- [ ] Keep the sub exposure the SeeStar defaults to (30 s) unless experimenting.
- [ ] Note the conditions (moon, transparency) for later triage of bad subs.

## After the session
- [ ] Export/transfer the **individual FITS subs** off the device (app export or
      storage). Drop them in this repo's `data/` (git-ignored) — e.g.
      `data/M101_subs_YYYYMMDD/`.
- [ ] Also grab the final stack for a quick-look comparison.

## Next: process
1. **Re-stack the subs** with sigma-clip rejection (Siril workflow in `siril/`) →
   clean master (removes satellite trails, lowers noise floor; optionally drizzle).
2. Run that master through the Python pipeline: `make run FITS="<master>.fit" V=<label>`
   (crop → background → **Gaia PCC** → stretch → denoise → finish).
3. Optional finishing: star reduction/separation (StarNet++ / SetiAstroSuitePro),
   deconvolution/sharpening on the galaxy, tighter crop.
