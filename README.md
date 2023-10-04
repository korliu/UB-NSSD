# UB-NSSD
Non-Sound Speech Detection (NSSD) Team for UB Voice.

## Development Environment
### Virtual Environment
1. `python -m venv .venv` to create virtual environment
2. [Activate virtual environment](https://docs.python.org/3/library/venv.html#how-venvs-work)
3. [Install `ffmpeg`](https://ffmpeg.org/download.html)
4. `pip install .` to install dependencies
5. `pip install --no-deps whisper-at` [(Windows/Mac only)](https://github.com/YuanGongND/whisper-at#step-1-install-whisper-at)
6. `deactivate` to leave virtual environment

### Nix Flake
1. `nix develop`

## Resources
* [`whisper-at`](https://github.com/YuanGongND/whisper-at)
* [`YAMNet`](https://tfhub.dev/google/yamnet/1)