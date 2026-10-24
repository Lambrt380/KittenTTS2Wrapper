import json
import re
import shutil
import sys
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PyQt6.QtCore import QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QTextCursor
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from voice_quality import rate_voice

VOICES_DIR = Path(__file__).resolve().parent / "voices"

EMOTIONS = [
    "mundane",
    "nervous",
    "tender",
    "angry",
    "excited",
    "stern",
    "sad",
    "contemplative",
    "surprised",
    "joyful",
]
EVENTS = [
    "pause",
    "sigh",
    "gasp",
    "laugh",
    "giggle",
    "sob",
    "scoff",
    "growl",
    "um",
    "gulp",
]
EMOTION_PREFIX = re.compile(
    r"^\s*\[(?:" + "|".join(EMOTIONS) + r")\]\s*", re.IGNORECASE
)


def load_builtin_voices():
    try:
        manifest = Path(
            hf_hub_download(
                "KittenML/kitten-tts-2",
                "voices/voices.json",
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        manifest = Path(__file__).with_name("builtin_voices.json")
        return {name: {"reference": ""} for name in json.loads(manifest.read_text())}
    return {
        name: {**spec, "reference": str(manifest.parent / spec["reference"])}
        for name, spec in json.loads(manifest.read_text()).items()
    }


class RenderThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    saved = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.voice = None
        self.reference_versions = {}

    def run(self):
        try:
            import numpy as np
            import soundfile as sf
            from kittenml import KittenTTS

            if self.reference:
                info = Path(self.reference).stat()
                version = (info.st_mtime_ns, info.st_size)
                previous = self.reference_versions.get(self.reference)
                if previous is not None and previous != version:
                    # KittenTTS caches references by path, so replaced WAVs need a fresh model.
                    self.log.emit("Reference recording changed; reloading the model…")
                    self.model = None
                    self.reference_versions.clear()
                self.reference_versions[self.reference] = version
            if self.model is None:
                self.status.emit("Loading model…")
                self.log.emit("Loading KittenTTS 2 (first load may download weights)…")
                self.model = KittenTTS("KittenML/kitten-tts-2")
            self.log.emit(f"Model ready on {self.model.device}.")
            if self.reference:
                self.log.emit(f"Reference: {self.reference}")
                if not self.transcript:
                    self.log.emit(
                        "No transcript supplied; Whisper will transcribe the reference."
                    )
            else:
                self.log.emit(f"Voice: {self.voice}")
            self.original_reference = (
                self.model.voices[self.voice]["reference"]
                if self.voice
                else self.reference
            )
            options = {"preset": self.preset}
            if self.reference:
                options.update(
                    reference=self.reference, reference_text=self.transcript or None
                )
            else:
                options["voice"] = self.voice
            clips = []
            self.progress.emit(0)
            for number, line in enumerate(self.lines, 1):
                self.status.emit(f"Rendering line {number} of {len(self.lines)}…")
                self.log.emit(f"Rendering line {number}/{len(self.lines)}: {line}")
                clip = self.model.generate(line, **options)
                clips.append(clip)
                if number < len(self.lines):
                    clips.append(
                        np.zeros(int(self.model.sample_rate * 0.2), dtype=np.float32)
                    )
                self.log.emit(
                    f"Line {number} finished ({len(clip) / self.model.sample_rate:.1f}s audio)."
                )
                self.progress.emit(number)
            self.log.emit("Writing WAV…")
            self.status.emit("Saving WAV…")
            sf.write(self.output, np.concatenate(clips), self.model.sample_rate)
            self.log.emit(f"Saved: {self.output}")
            self.saved.emit(self.output)
        except Exception:  # noqa: BLE001
            # A failed render must report its traceback without taking down the GUI.
            self.log.emit(traceback.format_exc())


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KittenTTS 2")
        self.resize(780, 620)
        self.worker = RenderThread(self)
        self.worker.log.connect(self.append_log)
        self.worker.saved.connect(self.audio_saved)
        self.worker.finished.connect(self.render_finished)
        self.saved_path = ""
        self.saved_reference = ""
        self.builtin_voices = load_builtin_voices()
        VOICES_DIR.mkdir(exist_ok=True)
        self.voice_ratings = {}
        self.library_stamp = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.errorOccurred.connect(
            lambda _, message: self.append_log(f"Playback error: {message}")
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages)
        self.controls = QWidget()
        self.pages.addWidget(self.controls)
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)
        self.form = QFormLayout()

        library_row = QHBoxLayout()
        self.library = QComboBox()
        self.library.setMinimumContentsLength(24)
        self.library.currentIndexChanged.connect(self.select_voice)
        library_row.addWidget(self.library, 1)
        self.preview = QPushButton("Listen")
        self.preview.clicked.connect(self.preview_voice)
        library_row.addWidget(self.preview)
        add = QPushButton("Add WAV…")
        add.clicked.connect(self.add_voice)
        library_row.addWidget(add)
        folder = QPushButton("Open folder")
        folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(VOICES_DIR)))
        )
        library_row.addWidget(folder)
        self.form.addRow("Voice", library_row)
        self.voice_details = QLabel()
        self.voice_details.setWordWrap(True)
        self.voice_details.setObjectName("hint")
        self.voice_details.setToolTip(
            "Recording suitability estimate, not a prediction of voice quality. "
            "Checks length, sample rate, level, clipping and silence in the first 30s. "
            "Use clean speech from one person; noise, music and speaker count are not detected."
        )
        self.form.addRow("Recording score", self.voice_details)

        self.transcript = QLineEdit()
        self.transcript.setPlaceholderText(
            "Optional: what the reference clip says (skips Whisper)"
        )
        self.form.addRow("Transcript", self.transcript)
        self.preset = QComboBox()
        self.preset.addItems(["expressive", "stable"])
        self.form.addRow("Delivery", self.preset)
        controls.addLayout(self.form)
        controls.addWidget(QLabel("Text — one spoken line per paragraph"))
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(
            "[joyful] Hello! <laugh> This is (((wonderful))).\n[contemplative] What should we say next?"
        )
        controls.addWidget(self.text, 1)

        tags = QHBoxLayout()
        self.emotion = QComboBox()
        self.emotion.addItems(EMOTIONS)
        self.emotion.setCurrentText("joyful")
        tags.addWidget(self.emotion)
        apply_emotion = QPushButton("Set line emotion")
        apply_emotion.clicked.connect(self.set_emotion)
        tags.addWidget(apply_emotion)
        self.event = QComboBox()
        self.event.addItems(EVENTS)
        tags.addWidget(self.event)
        insert_event = QPushButton("Insert event")
        insert_event.clicked.connect(self.insert_event)
        tags.addWidget(insert_event)
        emphasis = QPushButton("Emphasize selection")
        emphasis.clicked.connect(self.emphasize)
        tags.addWidget(emphasis)
        controls.addLayout(tags)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output WAV"))
        self.output = QLineEdit(str(Path.cwd() / "output.wav"))
        output_row.addWidget(self.output)
        save_as = QPushButton("Save as…")
        save_as.clicked.connect(self.choose_output)
        output_row.addWidget(save_as)
        controls.addLayout(output_row)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.hide()
        controls.addWidget(self.notice)

        buttons = QHBoxLayout()
        self.render = QPushButton("Render voice")
        self.render.clicked.connect(self.start_render)
        buttons.addStretch()
        buttons.addWidget(self.render)
        controls.addLayout(buttons)

        self.render_page = QWidget()
        self.pages.addWidget(self.render_page)
        rendering = QVBoxLayout(self.render_page)
        rendering.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("Preparing…")
        self.worker.status.connect(self.status.setText)
        rendering.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.worker.progress.connect(self.update_progress)
        rendering.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        rendering.addWidget(self.log, 1)
        self.comparison = QWidget()
        comparison = QVBoxLayout(self.comparison)
        comparison.setContentsMargins(0, 0, 0, 0)
        self.comparison_note = QLabel()
        self.comparison_note.setWordWrap(True)
        comparison.addWidget(self.comparison_note)
        playback = QHBoxLayout()
        self.original = QPushButton("Play original")
        self.original.setEnabled(False)
        self.original.clicked.connect(lambda: self.play_file(self.saved_reference))
        playback.addWidget(self.original)
        self.open_audio = QPushButton("Play result")
        self.open_audio.setEnabled(False)
        self.open_audio.clicked.connect(lambda: self.play_file(self.saved_path))
        playback.addWidget(self.open_audio)
        stop = QPushButton("Stop audio")
        stop.clicked.connect(self.player.stop)
        playback.addWidget(stop)
        comparison.addLayout(playback)
        rendering.addWidget(self.comparison)
        self.comparison.hide()
        self.back = QPushButton("Back to editing")
        self.back.clicked.connect(self.back_to_editing)
        rendering.addWidget(self.back)
        self.refresh_library()
        self.library_timer = QTimer(self)
        self.library_timer.timeout.connect(self.refresh_library)
        self.library_timer.start(2000)

    def refresh_library(self):
        if self.worker.isRunning():
            return
        selection = self.library.currentData() or "Bruno"
        clips = sorted(
            (
                path
                for path in VOICES_DIR.iterdir()
                if path.suffix.lower() == ".wav" and path.is_file()
            ),
            key=lambda path: path.name.lower(),
        )
        stamp = []
        for path in clips:
            try:
                info = path.stat()
                stamp.append((str(path), info.st_mtime_ns, info.st_size))
            except OSError:
                stamp.append((str(path), 0, 0))
        if stamp == self.library_stamp:
            return
        self.library_stamp = stamp
        self.library.blockSignals(True)
        self.library.clear()
        for name in sorted(self.builtin_voices):
            self.library.addItem(f"{name} · built-in", name)
        if stamp:
            self.library.insertSeparator(self.library.count())
        for path, modified, size in stamp:
            cached = self.voice_ratings.get(path)
            if cached is None or cached[0] != (modified, size):
                cached = ((modified, size), rate_voice(path))
                self.voice_ratings[path] = cached
            score, details = cached[1]
            stars = "★" * score + "☆" * (5 - score) if score else "unreadable"
            self.library.addItem(f"{Path(path).name} · {stars}", path)
            item = self.library.model().item(self.library.count() - 1)
            item.setToolTip(details)
            item.setEnabled(score > 0)
        index = self.library.findData(selection)
        if index < 0:
            index = self.library.findData("Bruno")
            self.transcript.clear()
        self.library.setCurrentIndex(index)
        self.library.blockSignals(False)
        self.update_voice_details()

    def update_voice_details(self):
        selection = self.library.currentData()
        custom = selection not in self.builtin_voices
        self.form.setRowVisible(self.voice_details, custom)
        self.form.setRowVisible(self.transcript, custom)
        if custom:
            score, details = self.voice_ratings[selection][1]
            self.voice_details.setText(f"{score}/5 · {details}")
            self.preview.setEnabled(score > 0)
        else:
            self.preview.setEnabled(bool(self.voice_reference()))

    def select_voice(self, _):
        self.transcript.clear()
        self.update_voice_details()

    def voice_reference(self):
        selection = self.library.currentData()
        if selection in self.builtin_voices:
            path = self.builtin_voices[selection]["reference"]
            return path if path and Path(path).is_file() else ""
        return selection

    def preview_voice(self):
        self.play_file(self.voice_reference())

    def add_voice(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add a reference voice (5–30 seconds)",
            str(VOICES_DIR),
            "WAV (*.wav *.WAV)",
        )
        if not path:
            return
        source = Path(path).resolve()
        target = VOICES_DIR / source.name
        if source != target:
            number = 2
            while target.exists():
                target = VOICES_DIR / f"{source.stem} ({number}){source.suffix}"
                number += 1
            try:
                shutil.copy2(source, target)
            except OSError as error:
                self.append_log(f"Could not add the voice: {error}")
                return
        self.refresh_library()
        self.library.setCurrentIndex(self.library.findData(str(target)))

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Output WAV", self.output.text(), "WAV (*.wav)"
        )
        if path:
            self.output.setText(
                path if path.lower().endswith(".wav") else path + ".wav"
            )

    def set_emotion(self):
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor
        )
        line = EMOTION_PREFIX.sub("", cursor.selectedText())
        cursor.insertText(f"[{self.emotion.currentText()}] {line}")
        self.text.setTextCursor(cursor)
        self.text.setFocus()

    def insert_event(self):
        self.text.insertPlainText(f" <{self.event.currentText()}> ")
        self.text.setFocus()

    def emphasize(self):
        cursor = self.text.textCursor()
        if cursor.hasSelection() and "\u2029" not in cursor.selectedText():
            cursor.insertText(f"((({cursor.selectedText()})))")
            self.text.setFocus()
        else:
            self.append_log("Select words within one line to emphasize them.")

    def start_render(self):
        lines = [
            line.strip()
            for line in self.text.toPlainText().splitlines()
            if line.strip()
        ]
        selection = self.library.currentData()
        reference = "" if selection in self.builtin_voices else selection
        output = self.output.text().strip()
        if not lines:
            self.append_log("Enter some text first.")
            return
        if reference and not Path(reference).is_file():
            self.append_log("Reference audio does not exist.")
            return
        if not output or Path(output).suffix.lower() != ".wav":
            self.append_log("Choose an output filename ending in .wav.")
            return
        output = str(Path(output).resolve())
        original = self.voice_reference()
        if original and Path(original).resolve() == Path(output):
            self.append_log(
                "Choose a different output file to preserve your reference audio."
            )
            return
        if not Path(output).parent.is_dir():
            self.append_log("The output folder does not exist.")
            return
        self.worker.lines = lines
        self.worker.voice = selection if selection in self.builtin_voices else None
        self.worker.reference = str(Path(reference).resolve()) if reference else ""
        self.worker.transcript = self.transcript.text().strip() if reference else ""
        self.worker.preset = self.preset.currentText()
        self.worker.output = output
        self.log.clear()
        self.notice.hide()
        self.saved_path = ""
        self.saved_reference = ""
        self.comparison_note.clear()
        self.comparison.hide()
        self.status.setText("Preparing…")
        self.progress.setRange(0, 0)
        self.controls.setEnabled(False)
        self.open_audio.setEnabled(False)
        self.original.setEnabled(False)
        self.back.setEnabled(False)
        self.back.hide()
        self.player.stop()
        self.player.setSource(QUrl())
        self.pages.setCurrentWidget(self.render_page)
        self.append_log("Starting render…")
        self.worker.start()

    def append_log(self, message):
        self.log.appendPlainText(message)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if self.pages.currentWidget() == self.controls:
            self.notice.setText(message)
            self.notice.show()

    def update_progress(self, number):
        self.progress.setRange(0, len(self.worker.lines))
        self.progress.setValue(number)

    def audio_saved(self, path):
        self.saved_path = path
        self.saved_reference = self.worker.original_reference
        reference_name = self.worker.voice or Path(self.saved_reference).name
        self.comparison_note.setText(
            f"Original: {reference_name}\nResult: {Path(path).name}"
        )

    def render_finished(self):
        self.controls.setEnabled(True)
        self.open_audio.setEnabled(bool(self.saved_path))
        self.original.setEnabled(bool(self.saved_reference))
        self.back.setEnabled(True)
        self.back.show()
        self.comparison.setVisible(bool(self.saved_path))
        self.status.setText(
            "Finished" if self.saved_path else "Render failed — see log below"
        )
        if not self.saved_path:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        self.builtin_voices = load_builtin_voices()
        self.library_stamp = None
        self.refresh_library()

    def back_to_editing(self):
        self.player.stop()
        self.pages.setCurrentWidget(self.controls)
        self.notice.hide()

    def play_file(self, path):
        if not path or not Path(path).is_file():
            self.append_log("Audio file does not exist.")
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
        self.player.play()
        self.append_log(f"Playing: {Path(path).name}")

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.append_log("Wait for the render to finish before closing.")
            event.ignore()
        else:
            self.player.stop()
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { background: #f3f3f3; color: #333; font-size: 13px; }
        QLabel#hint { color: #777; }
        QLineEdit, QPlainTextEdit, QComboBox {
            background: white; border: 1px solid #ccc; border-radius: 3px; padding: 5px;
        }
        QPushButton { background: #e9e9e9; border: 1px solid #c5c5c5; border-radius: 3px; padding: 6px 10px; }
        QPushButton:hover { background: #ddd; }
        QPushButton:disabled { color: #999; }
        QProgressBar { border: 1px solid #ccc; text-align: center; }
        QProgressBar::chunk { background: #aaa; }
    """)
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
