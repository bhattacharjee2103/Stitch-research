#!/bin/zsh
# refcam launcher — double-click this file in Finder.
cd "$(dirname "$0")" || exit 1

if [[ ! -x .venv/bin/python ]]; then
  print "refcam environment is missing."
  print "Run: cd $(pwd) && .venv/bin/pip install -r requirements-dev.txt"
  read "?Press Return to close..."
  exit 1
fi

print ""
print "refcam"
print "  1) Live camera preview"
print "  2) Add a reference from the live camera"
print "  3) Add a reference from a photo file"
print "  4) List references"
print "  5) Attach an overlay picture"
print "  6) Remove a reference"
print "  7) Run tests"
print "  8) Quit"
print ""
read "?Choose [1-8]: " choice

case "$choice" in
  1)
    exec .venv/bin/python -m refcam run
    ;;
  2)
    .venv/bin/python -m refcam add
    ;;
  3)
    read "?Photo path: " photo
    .venv/bin/python -m refcam add --image "$photo"
    ;;
  4)
    .venv/bin/python -m refcam list
    ;;
  5)
    read "?Reference name: " name
    read "?Overlay image path: " image
    .venv/bin/python -m refcam overlay "$name" "$image"
    ;;
  6)
    read "?Reference name: " name
    .venv/bin/python -m refcam remove "$name"
    ;;
  7)
    .venv/bin/python -m refcam test
    ;;
  *)
    exit 0
    ;;
esac

print ""
read "?Press Return to close..."
