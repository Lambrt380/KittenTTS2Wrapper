### kitten tts 2 qt6 wrapper

**wrapper for kitten tts 2 written in qt6**

simple wrapper(a program wrapper, for easier use of kitten tts 2) for a local "neural network" text to speech model

## dependencies(check requirement.txt):
kittenml==0.9.4  
PyQt6>=6.7,<7  
soundfile>=0.12  
json  
FFmpeg (optional, needed for checking audio recordings, you can use the program without it)

### **big warning**
the program was tested on NixOs Yarara 26.02 ke 6.18.39 using "shell.nix" and a wayland compositor, nowhere else, the program might not work on windows because of different syntax/paths

## running:
python -m venv .venv
source .venv/bin/activate | .venv/bin/activate.fish  
.venv/bin/python -m pip install -r requirements  
nix-shell --run '.venv/bin/python app.py' | .venv/bin/python app.py
