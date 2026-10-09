# Facial Nerve Palsy – Bell's palsy simulation

A tablet app for a neuroscience exhibition (designed for a Samsung Galaxy Tab
S9 FE held upright). The front camera shows the visitor's face with one side
changed to look like **Bell's palsy** (peripheral facial nerve palsy).

*Simulation for education. Not a diagnosis.*

## What the visitor sees

| Step | Time |
| --- | --- |
| Countdown **3 – 2 – 1** | 3 s |
| Title **Facial Nerve Palsy** | 4 s |
| Live palsy mirror: "Try to smile", "Raise your eyebrows", "Close your eyes tight" | 6 s each |
| 📸 One photo (face-position check, then 3-2-1) | about 6 s |
| Drooping eyebrow (on the photo) | 5 s |
| Eye cannot close fully | 5 s |
| Flattened cheek and smile line | 5 s |
| Drooping mouth corner | 5 s |
| All symptoms together: normal face and Bell's palsy face | 5 s |
| **Thank you for the experience** | 10 s, then back to the start |

Each symptom fades in on the photo, stays, then fades out before the next one.

## Controls on the live screen

- **Left side / Right side**: which side of the face is affected.
- **Severity**: House-Brackmann grade I (normal) to VI (total paralysis). This scales all movements.
- **Show original**: switch the effect off and on.
- **Take photo**: skip the prompts and go straight to the photo.
- **Double-tap** the picture at any time to return to the start screen.

## How the simulation works (for the curious)

- **Google MediaPipe Face Landmarker** finds 478 points on the face in every frame.
- The points are joined into triangles (**Delaunay triangulation**). WebGL then
  draws the camera image through the moved triangles, which is a triangle-mesh warp.
- Only points on the affected side of the **facial midline** (landmarks
  10, 168, 6, 1, 152) move. The other side, the hair, the neck and the background
  stay untouched.
- Moves use small distances with a soft Gaussian falloff (sigma 30–40 px):
  - **Brow**: lowered, with forehead wrinkles smoothed.
  - **Eye**: looks wider, with a slight lower-lid sag and a faint lower-lid shadow.
    The iris is never stretched.
  - **Cheek**: mild sag, with the nasolabial fold smoothed.
  - **Mouth**: the corner is pulled down 8–15 px and slightly outward, and the upper lip on that side is flattened.
- **Reacts to the visitor's face**:
  - When the visitor smiles, the affected side moves only 20–40% as much.
  - When they raise their eyebrows, the affected brow doesn't move.
  - When they close their eyes, the affected eye stays slightly open.
- Point jitter is smoothed with an exponential moving average.

Each step is explained with comments in `app.js`.

## Put it online (GitHub Pages)

Settings → Pages → *Deploy from a branch* → choose the branch with this app and `/ (root)` → Save.
The address will look like `https://axisavalon33-ai.github.io/Facial-palsy-/`.

## Changing timings

Open `app.js` and change the numbers in `CONFIG` at the top. Add `?quick` to
the address for a fast preview.

## Files

- `index.html`, `style.css`: the screens and buttons
- `app.js`: everything else (tracking, mesh warp, timing)
- `lib/`, `models/`: a backup copy of MediaPipe (Apache 2.0), used only if the CDN can't be reached

No photos or video leave the tablet.
