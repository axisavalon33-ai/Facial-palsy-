# Facial Nerve Palsy – tablet experience

An app for a neuroscience exhibition, made to run on a **tablet**. The
**front camera** takes a photo of the visitor, then one side of their face in
the photo is changed to show a symptom of **Bell's palsy** (facial nerve palsy).
A new photo is taken for every symptom.

## What the visitor sees

| Step | Time |
| --- | --- |
| Countdown **3 – 2 – 1** | 3 s |
| Title **Facial Nerve Palsy** | 4 s |
| 📸 Photo 1 ("look at the camera" 3-2-1, flash) | 3 s |
| Symptom 1: Drooping eyebrow (your face next to the changed face) | 30 s |
| 📸 Photo 2 | 3 s |
| Symptom 2: Eye cannot close fully (watery eye) | 30 s |
| 📸 Photo 3 | 3 s |
| Symptom 3: Flattened cheek and smile line | 30 s |
| 📸 Photo 4 | 3 s |
| Symptom 4: Drooping mouth corner | 30 s |
| 📸 Photo 5 | 3 s |
| Symptom 5: Drooling | 30 s |
| 📸 Final photo | 3 s |
| **All symptoms together**: normal face next to the Bell's palsy face | 5 s |
| **Thank you for the experience** | 15 s, then back to the start screen |

If no face is visible when a photo is due, the app waits until the person puts
their face inside the oval.

## How to put it online (one time only)

1. Open this repository on github.com.
2. Click **Settings** (top bar), then **Pages** (left menu).
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Under **Branch**, pick the branch that holds this app
   (`main`, or `claude/bells-palsy-vr-app-9o6f2t`), folder **/ (root)**, then **Save**.
5. Wait 1–2 minutes and refresh. GitHub shows the address:
   `https://axisavalon33-ai.github.io/Facial-palsy-/`.

The repository must be **public** for free GitHub Pages.

## How to use it at the exhibition

1. Open the address on the tablet in **Chrome** (Android) or **Safari** (iPad).
2. Tap **Start** and allow the camera.
3. The visitor stands or sits in front of the tablet and looks at the screen.
4. **Double-tap** the screen at any time to go back to the start screen.

Under **Settings** on the start screen you can choose which side of the face is
affected and how strong the effect is.

Tips: good light on the face, the tablet at face height (a stand helps),
keep it plugged in.

Add `?quick` to the end of the address to preview everything quickly
(5 s per symptom).

## Changing the timings

Open `app.js`, find `CONFIG` near the top and change the numbers (seconds).
On github.com: click the file, then the ✏️ pencil icon, then **Commit changes**.

## Files

- `index.html`, `style.css`: the start screen
- `app.js`: the experience (face tracking, photos, effects, timing)
- `lib/`: Google MediaPipe face tracker (Apache 2.0 licence)
- `models/face_landmarker.task`: the face-tracking model

Everything runs inside the browser. No photo or video is uploaded or saved anywhere.
