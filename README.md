<div align="center">

<h1>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/25470747/280868566-d4e8ebf4-34d1-49cd-9db5-607e8eb24a20.png">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/25470747/280868788-bdea8968-7139-46b4-90b3-0d81d0fe81b7.png">
  <img alt="NSSD" height="128" src="https://user-images.githubusercontent.com/25470747/280868788-bdea8968-7139-46b4-90b3-0d81d0fe81b7.png">
</picture>
</h1>

</div>

Non-Sound Speech Detection (NSSD) Team for University at Buffalo Voice.

## Development Environment
### Virtual Environment
1. `python -m venv .venv` to create virtual environment
2. [Activate virtual environment](https://docs.python.org/3/library/venv.html#how-venvs-work)
3. [Install `ffmpeg`](https://ffmpeg.org/download.html)
4. `pip install .` to install dependencies
5. `deactivate` to leave virtual environment
> [!IMPORTANT]
> As of 10/30/2023, the latest versions of `tensorflow_io` do not ship with Windows builds. You must manually compile it, try an older version, or use a different OS.

### Nix Flake
1. `nix develop`
> [!NOTE]
> Python packages in the flake are not pinned, so there may be version incompatabilities in the future. Please refer to the `pyproject.toml` for supported dependency versions.


## Resources
* [`whisper-at`](https://github.com/YuanGongND/whisper-at)
* [`YAMNet`](https://tfhub.dev/google/yamnet/1)
