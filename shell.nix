{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    libglvnd
    libxkbcommon
    dbus
    fontconfig
    freetype
    glib
    zlib
    zstd
    libpulseaudio
    alsa-lib
    libkrb5
    brotli
    bzip2
    libdrm
    pipewire
    libxrandr
    libxext
    libx11
    libxcb
    libxcb-util
    libxcb-cursor
    libxcb-image
    libxcb-keysyms
    libxcb-render-util
    libxcb-wm
  ]);
  QT_QPA_PLATFORM = "xcb";
}
