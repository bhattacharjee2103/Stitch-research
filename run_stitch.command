#!/bin/zsh
cd "$(dirname "$0")"
export TK_SILENCE_DEPRECATION=1
export HF_HOME="$PWD/.cache/huggingface"
if [[ ! -x .venv/bin/python ]]; then
  print "Stitch environment is missing. Run: bash scripts/setup.sh"
  read "?Press Return to close..."
  exit 1
fi
exec .venv/bin/python -m stitch.app --camera FaceTime --connect "$@"
