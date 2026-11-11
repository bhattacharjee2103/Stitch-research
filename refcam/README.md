# refcam

Reference-object overlay on a live camera. A webcam frame is frozen, a region
is selected, and that crop becomes a reference. When the camera sees it again,
a detector verifies it geometrically and an overlay is drawn on it.

Runtime dependencies: NumPy and OpenCV only. Python 3.11.

## What it does

Save a textured, mostly flat object as a reference. In live camera, video, or
image mode, refcam uses SIFT feature matching and a RANSAC homography to find
that object. It draws the detected outline, or projects an attached image onto
the object with perspective correction. All processing happens locally.

## Setup

```sh
cd refcam
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

## Run

```sh
.venv/bin/python -m refcam run                  # default camera 0
.venv/bin/python -m refcam run --camera 1
.venv/bin/python -m refcam run --video clip.mp4
.venv/bin/python -m refcam run --image photo.jpg
```

The preview shows the keyboard shortcuts:

- Press `c` to freeze the current camera frame, drag a box around the object,
  press Enter, then type a reference name in the Terminal. It is saved and
  detection starts immediately; the camera stays open.
- Press `p` to attach an image to a saved reference. Enter the reference name
  and image path in the Terminal. The preview switches to project style.
- Press `o` to cycle between `outline` and `project` styles.
- Press `q` or `Esc` to quit.

## References

```sh
.venv/bin/python -m refcam add mug               # scan, SPACE freezes, draw box
.venv/bin/python -m refcam add mug --image photo.jpg
.venv/bin/python -m refcam list
.venv/bin/python -m refcam overlay mug sticker.png
.venv/bin/python -m refcam remove mug            # asks for confirmation
```

Each reference lives in `refs/<name>/` with `crop.png`, `meta.json`,
`features.npz` and an optional `overlay.png`. A reference needs at least 40
SIFT keypoints; otherwise the add flow asks for a larger or more detailed
region.

## Everyday use without an AI

Double-click **`refcam.command`** in Finder and pick a menu item. That is all
you need for day-to-day use.

The same thing from Terminal:

```sh
cd /Users/bhattacharjee/Desktop/internship/refcam
.venv/bin/python -m refcam run                     # press C to save a reference
.venv/bin/python -m refcam list                    # show saved references
```

While `run` is open, press `p` and follow the Terminal prompts to attach an
image. Do not type another shell command until you quit the preview.

The separate `add` and `overlay` commands are also available if you prefer to
select a photo from disk or attach an image outside the live preview.

Camera permission: the first time, macOS asks whether Terminal may use the
camera — click Allow. To change it later: System Settings → Privacy &
Security → Camera.

Matching works best for textured, rigid, flat objects viewed at a moderate
angle. Plain, glossy, transparent, deformable or heavily rotated objects can
be difficult to detect. Use `--video` or `--image` to try a saved reference
without a webcam.

## Tests

```sh
.venv/bin/python -m refcam test
```

## Extending

See [docs/ADDING.md](docs/ADDING.md). New detectors, overlays and commands are
one file each; core files never need editing.

