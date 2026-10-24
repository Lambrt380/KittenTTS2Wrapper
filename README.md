### kitten tts 2 qt6 wrapper

**wrapper for kitten tts 2 written in qt6**

simple wrapper(a program wrapper, for easier use of kitten tts 2) for a local "neural network" text to speech model

## dependencies

Python 3.10+ and the packages in `requirements.txt`:

- kittenml==0.9.4
- PyQt6>=6.7,<7
- soundfile>=0.12

FFmpeg is optional for audio support. `json` is part of Python.

### **big warning**
the program was tested on NixOs Yarara 26.02 ke 6.18.39 using "shell.nix" and a wayland compositor, nowhere else, the program might not work on windows because of different syntax/paths

## running

Create your own environment after cloning; the repository does not include one:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

On NixOS, run the same commands inside `nix-shell`, or launch an existing environment with:

```sh
nix-shell --run '.venv/bin/python app.py'
```

The first render downloads the model from Hugging Face. Dependencies and model downloads can use several GB of disk space. The voice list opens without downloading the model; built-in previews become available once their audio is cached.

Add your own reference WAV files through the app or put them in `voices/`. Recordings, rendered WAVs, virtual environments, model weights and local credentials are ignored by Git.

## tests

With dependencies installed:

```sh
QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

On NixOS, run this inside `nix-shell`. Tests use synthetic audio and a mocked model; they do not download weights or verify real model inference.
