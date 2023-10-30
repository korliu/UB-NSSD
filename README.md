# UB-NSSD
Non-Sound Speech Detection (NSSD) Team for UB Voice.

## Development Environment
> [!IMPORTANT]
> As of 10/30/2023, the latest versions of `tensorflow` do not ship with Windows builds. You must manually compile it, try an older version, or use a different OS.

### Virtual Environment
1. `python -m venv .venv` to create virtual environment
2. [Activate virtual environment](https://docs.python.org/3/library/venv.html#how-venvs-work)
3. [Install `ffmpeg`](https://ffmpeg.org/download.html)
4. `pip install .` to install dependencies
5. `deactivate` to leave virtual environment

### Nix Flake
1. `nix develop`

## Resources
* [`whisper-at`](https://github.com/YuanGongND/whisper-at)
* [`YAMNet`](https://tfhub.dev/google/yamnet/1)
