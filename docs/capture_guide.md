# Phone Capture Guide

The pipeline tolerates imperfect input, but good capture makes the difference
between a great bust and a melted blob. Follow these rules.

## The ideal capture

- **Short video > many photos.** One slow 360° walkaround (~30–60 s) gives the
  photogrammetry path its best chance.
- **Lock exposure and focus** (tap-and-hold on the subject until "AE/AF lock").
  Auto-exposure flicker breaks reconstruction.
- **Even lighting.** Soft, diffuse light; avoid mixed warm/cool sources and
  strong window light. No direct flash — skin gets shiny.
- **Walk slowly and steadily** around the subject, keeping the phone level,
  at roughly chest height. Cover the full circle *plus* a pass slightly above
  and below the head.
- **Capture the top of the head and the chin** — most busts fail here.
- **Subject stays still.** Head-and-shoulders pose, no smiling if possible
  (teeth confuse reconstruction), hair pulled back if practical.

## Avoid

- Busy, moving backgrounds (people walking behind the subject).
- Shiny skin / glossy makeup / wet hair — specular highlights break matching.
- Glasses (reflections) and hats that cover the face.
- Extreme close-ups with motion blur; keep the subject fully in frame with
  margin around the head.
- Zooming while recording. Use your feet.

## For the single-portrait fallback

Any decent sharp frontal photo works — TripoSR needs just one good, centered,
well-lit portrait with the subject filling ~40–80 % of the frame. A plain
background helps the background-removal stage (rembg) cut the subject out
cleanly.

## After capture

- Prefer original-resolution files; don't re-compress or resize.
- Phone photos are auto-rotated by the pipeline (EXIF orientation), so no
  manual rotation needed.
- Keep videos under ~2 minutes; longer recordings just produce more
  near-duplicate frames that get filtered anyway.
