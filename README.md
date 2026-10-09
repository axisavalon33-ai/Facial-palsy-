# Facial Nerve Palsy – VR experience

A live camera experience for a neuroscience exhibition. The camera films a face
and the app changes one side of it to look like **Bell's palsy** (facial nerve
palsy), one symptom at a time.

## What the visitor sees

| Step | Time |
| --- | --- |
| "Put the phone in the headset" (VR mode only) | 6 s |
| Countdown **3 – 2 – 1** | 3 s |
| Title **Facial Nerve Palsy** | 4 s |
| Symptom 1 – Drooping eyebrow | 30 s |
| Symptom 2 – Eye cannot close fully (watery eye) | 30 s |
| Symptom 3 – Flattened cheek and smile line | 30 s |
| Symptom 4 – Drooping mouth corner | 30 s |
| Symptom 5 – Drooling | 30 s |
| All symptoms together (live) | 30 s |
| Snapshot: normal face next to Bell's palsy face | 5 s |
| **Thank you for the experience** | 15 s, then back to the menu |

## How to put it online (one time only)

1. Open this repository on github.com.
2. Click **Settings** (top bar) → **Pages** (left menu).
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Under **Branch**, pick the branch that holds this app
   (`main`, or `claude/bells-palsy-vr-app-9o6f2t`), folder **/ (root)**, then **Save**.
5. Wait 1–2 minutes and refresh. GitHub shows the address, for example
   `https://axisavalon33-ai.github.io/Facial-palsy-/`.

The repository must be **public** for free GitHub Pages.

## How to use it at the exhibition

**With a VR headset (phone headset such as Google Cardboard / VR Box):**

1. Open the address in **Chrome** (Android) or **Safari** (iPhone).
2. Tap **Start VR** and allow the camera.
3. Put the phone in the headset (you have 6 seconds).
4. The **back camera** is used, so the visitor should face a **mirror** (to see
   their own face) or look at **another person**. The headset's front cover must
   not block the phone's camera.

**Without a headset (laptop, tablet, or TV with a webcam):**
tap **Start on screen**. The front (selfie) camera is used and the visitor sees
themselves like in a mirror.

**Double-tap** the screen at any time to return to the menu.
Under **Settings** in the menu you can choose the camera, the affected side,
and how strong the effect is.

Add `?quick` to the end of the address to preview everything in about a minute
(5 s per symptom).

## Changing the timings

Open `app.js`, find `CONFIG` near the top and change the numbers (seconds).
On github.com you can do this by clicking the file, then the ✏️ pencil icon,
then **Commit changes**.

## Files

- `index.html`, `style.css` – the start menu
- `app.js` – the experience (face tracking, effects, timeline)
- `lib/` – Google MediaPipe face tracker (Apache 2.0 licence)
- `models/face_landmarker.task` – the face-tracking model

Everything runs inside the browser; no video is uploaded or stored anywhere.
