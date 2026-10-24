import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from huggingface_hub.errors import LocalEntryNotFoundError
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QTextCursor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel

from app import Window, load_builtin_voices
from voice_quality import rate_voice


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.voices = Path(self.folder.name) / "voices"
        self.voice_folder_patch = patch("app.VOICES_DIR", self.voices)
        self.voice_folder_patch.start()
        self.addCleanup(self.voice_folder_patch.stop)
        self.window = Window()
        self.addCleanup(self.window.close)
        self.addCleanup(self.window.library_timer.stop)
        self.addCleanup(self.window.player.setSource, QUrl())
        self.output = str(Path(self.folder.name) / "voice.wav")
        self.window.output.setText(self.output)

    def select_clip(self, path):
        self.window.refresh_library()
        self.window.library.setCurrentIndex(self.window.library.findData(str(path)))

    def test_editor_starts_without_header_reference_bar_or_render_log(self):
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.pages.currentWidget(), self.window.controls)
        self.assertTrue(self.window.text.isVisible())
        self.assertFalse(self.window.log.isVisible())
        self.assertFalse(self.window.transcript.isVisible())
        labels = [label.text() for label in self.window.findChildren(QLabel)]
        self.assertNotIn("KittenTTS 2", labels)
        self.assertNotIn("Reference voice", labels)
        self.assertEqual(len(self.window.builtin_voices), 47)
        for name in self.window.builtin_voices:
            self.assertGreaterEqual(self.window.library.findData(name), 0)

    def test_builtin_catalog_is_available_without_cached_model_or_downloads(self):
        with patch(
            "app.hf_hub_download", side_effect=LocalEntryNotFoundError("not cached")
        ) as download:
            voices = load_builtin_voices()
        self.assertEqual(len(voices), 47)
        self.assertIn("Russian", voices)
        self.assertTrue(all(not spec["reference"] for spec in voices.values()))
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_emotion_replaces_tag_on_current_line_without_merging_lines(self):
        self.window.text.setPlainText("First line.\n[sad] Second line.\nThird line.")
        cursor = self.window.text.textCursor()
        cursor.setPosition(22)
        self.window.text.setTextCursor(cursor)
        self.window.emotion.setCurrentText("joyful")
        self.window.set_emotion()
        self.assertEqual(
            self.window.text.toPlainText(),
            "First line.\n[joyful] Second line.\nThird line.",
        )

    def test_vocal_event_and_emphasis(self):
        self.window.text.setPlainText("Hello world")
        cursor = self.window.text.textCursor()
        cursor.setPosition(6)
        cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
        self.window.text.setTextCursor(cursor)
        self.window.emphasize()
        self.window.event.setCurrentText("laugh")
        self.window.insert_event()
        self.assertEqual(self.window.text.toPlainText(), "Hello (((world))) <laugh> ")

    def test_empty_text_and_reference_output_collision_are_rejected(self):
        with patch.object(self.window.worker, "start") as start:
            self.window.start_render()
            start.assert_not_called()
            self.assertIn("Enter some text", self.window.log.toPlainText())
            clip = self.voices / "reference.wav"
            sf.write(clip, np.ones(2400) * 0.1, 24000)
            self.window.output.setText(str(clip))
            self.select_clip(clip)
            self.window.text.setPlainText("Hello.")
            self.window.start_render()
            start.assert_not_called()
            self.assertIn("preserve your reference", self.window.log.toPlainText())
            self.assertEqual(self.window.pages.currentWidget(), self.window.controls)
            self.assertIn("preserve your reference", self.window.notice.text())

    def test_background_render_preserves_markup_and_saves_wav(self):
        model = Mock(sample_rate=24000, device="cpu")
        model.generate.return_value = np.ones(2400, dtype=np.float32) * 0.1
        self.window.worker.model = model
        self.window.text.setPlainText("[joyful] Hello <laugh>.\n\n(((Goodbye))).")
        reference = self.voices / "reference.wav"
        sf.write(reference, np.ones(2400) * 0.1, 24000)
        self.select_clip(reference)
        self.window.transcript.setText("Reference words.")
        self.window.start_render()
        self.assertEqual(self.window.pages.currentWidget(), self.window.render_page)
        self.assertFalse(self.window.back.isEnabled())
        self.assertTrue(self.window.comparison.isHidden())
        self.assertTrue(self.window.back.isHidden())
        self.assertEqual(self.window.progress.maximum(), 0)
        self.assertFalse(self.window.render.isEnabled())
        self.assertTrue(self.window.worker.wait(10000))
        self.app.processEvents()
        self.assertEqual(model.generate.call_count, 2)
        model.generate.assert_any_call(
            "[joyful] Hello <laugh>.",
            preset="expressive",
            reference=str(reference),
            reference_text="Reference words.",
        )
        audio, rate = sf.read(self.output)
        self.assertEqual(rate, 24000)
        self.assertEqual(len(audio), 9600)
        self.assertTrue(np.all(audio[2400:7200] == 0))
        self.assertEqual(self.window.saved_path, self.output)
        self.assertEqual(self.window.progress.value(), 2)
        self.assertTrue(self.window.render.isEnabled())
        self.assertTrue(self.window.open_audio.isEnabled())
        self.assertTrue(self.window.back.isEnabled())
        self.assertFalse(self.window.comparison.isHidden())
        self.assertEqual(self.window.status.text(), "Finished")
        self.window.back.click()
        self.assertEqual(self.window.pages.currentWidget(), self.window.controls)
        self.assertIn("(((Goodbye)))", self.window.text.toPlainText())

    def test_render_failure_is_logged_and_controls_recover(self):
        model = Mock(sample_rate=24000, device="cpu", voices=self.window.builtin_voices)
        model.generate.side_effect = RuntimeError("Rendering failed")
        self.window.worker.model = model
        self.window.text.setPlainText("Hello.")
        self.window.start_render()
        self.assertTrue(self.window.worker.wait(10000))
        self.app.processEvents()
        self.assertIn("RuntimeError: Rendering failed", self.window.log.toPlainText())
        self.assertTrue(self.window.render.isEnabled())
        self.assertFalse(self.window.open_audio.isEnabled())
        self.assertFalse(Path(self.output).exists())
        self.assertIn("Render failed", self.window.status.text())
        self.assertTrue(self.window.back.isEnabled())
        self.window.back.click()
        self.assertEqual(self.window.pages.currentWidget(), self.window.controls)

    def test_builtin_voice_uses_native_voice_argument_and_has_original_for_comparison(
        self,
    ):
        reference = Path(self.folder.name) / "builtin.wav"
        sf.write(reference, np.ones(2400) * 0.1, 24000)
        self.window.builtin_voices["Russian"]["reference"] = str(reference)
        model = Mock(sample_rate=24000, device="cpu", voices=self.window.builtin_voices)
        model.generate.return_value = np.ones(2400, dtype=np.float32) * 0.1
        self.window.worker.model = model
        self.window.library.setCurrentIndex(self.window.library.findData("Russian"))
        with patch.object(self.window, "play_file") as play:
            self.window.preview.click()
            play.assert_called_with(self.window.builtin_voices["Russian"]["reference"])
        self.window.transcript.setText("Should not be sent for built-in voices.")
        self.window.text.setPlainText("Hello.")
        self.window.start_render()
        self.assertTrue(self.window.worker.wait(10000))
        self.app.processEvents()
        model.generate.assert_called_once_with(
            "Hello.", preset="expressive", voice="Russian"
        )
        self.assertEqual(
            self.window.saved_reference, model.voices["Russian"]["reference"]
        )
        self.assertTrue(self.window.original.isEnabled())

    def test_recording_scores_and_unreadable_audio(self):
        path = self.voices / "sample.wav"
        tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(24000 * 8) / 24000)
        sf.write(path, tone, 24000)
        self.assertEqual(rate_voice(path)[0], 5)
        sf.write(path, np.zeros(24000 * 8), 24000)
        score, details = rate_voice(path)
        self.assertEqual(score, 1)
        self.assertIn("very quiet", details)
        self.assertIn("silence", details)
        sf.write(path, np.ones(24000 * 2), 24000)
        score, details = rate_voice(path)
        self.assertLessEqual(score, 2)
        self.assertIn("too short", details)
        self.assertIn("clipping", details)
        path.write_bytes(b"broken wav")
        self.assertEqual(rate_voice(path)[0], 0)

    def test_library_detects_added_changed_and_removed_wavs(self):
        clip = self.voices / "Voice.WAV"
        sf.write(clip, np.ones(24000) * 0.1, 24000)
        QTest.qWait(2200)
        index = self.window.library.findData(str(clip))
        self.assertGreater(index, 0)
        self.window.library.setCurrentIndex(index)
        self.assertEqual(self.window.voice_reference(), str(clip))
        self.assertIn("too short", self.window.voice_details.text())
        sf.write(clip, np.ones(24000 * 8) * 0.1, 24000)
        self.window.refresh_library()
        self.assertIn("5/5", self.window.voice_details.text())
        broken = self.voices / "broken.wav"
        broken.touch()
        self.window.refresh_library()
        index = self.window.library.findData(str(broken))
        self.assertFalse(self.window.library.model().item(index).isEnabled())
        clip.unlink()
        self.window.refresh_library()
        self.assertEqual(self.window.library.findData(str(clip)), -1)
        self.assertEqual(self.window.library.currentData(), "Bruno")

    def test_comparison_uses_render_reference_after_voice_selection_changes(self):
        original = self.voices / "original.wav"
        other = self.voices / "other.wav"
        sf.write(original, np.ones(2400) * 0.1, 24000)
        sf.write(other, np.ones(2400) * 0.2, 24000)
        sf.write(self.output, np.ones(2400) * 0.3, 24000)
        self.window.worker.original_reference = str(original)
        self.window.audio_saved(self.output)
        self.window.render_finished()
        self.select_clip(other)
        with patch.object(self.window, "play_file") as play:
            self.window.original.click()
            play.assert_called_with(str(original))
            self.window.open_audio.click()
            play.assert_called_with(self.output)
            self.window.preview.click()
            play.assert_called_with(str(other))

    def test_playback_switches_sources_and_missing_file_is_logged(self):
        self.assertTrue(self.window.player.isAvailable())
        sf.write(self.output, np.ones(2400) * 0.1, 24000)
        with (
            patch.object(self.window.player, "stop") as stop,
            patch.object(self.window.player, "play") as play,
            patch.object(self.window.player, "setSource") as source,
        ):
            self.window.play_file(self.output)
            source.assert_called_once_with(QUrl.fromLocalFile(self.output))
            stop.assert_called_once()
            play.assert_called_once()
            self.window.play_file(str(self.voices / "missing.wav"))
            play.assert_called_once()
            self.assertIn("Audio file does not exist", self.window.log.toPlainText())

    def test_replaced_reference_reloads_model_instead_of_reusing_cached_voice(self):
        reference = self.voices / "voice.wav"
        sf.write(reference, np.ones(2400) * 0.1, 24000)
        old_model = Mock()
        new_model = Mock(sample_rate=24000, device="cpu")
        new_model.generate.return_value = np.ones(2400, dtype=np.float32) * 0.1
        self.window.worker.model = old_model
        self.window.worker.reference_versions[str(reference)] = (0, 0)
        self.select_clip(reference)
        self.window.text.setPlainText("Hello.")
        with patch("kittenml.KittenTTS", return_value=new_model) as load:
            self.window.start_render()
            self.assertTrue(self.window.worker.wait(10000))
            self.app.processEvents()
            load.assert_called_once_with("KittenML/kitten-tts-2")
        old_model.generate.assert_not_called()
        self.assertEqual(self.window.worker.model, new_model)

    def test_add_wav_copies_into_library_without_overwriting_existing_recording(self):
        source = Path(self.folder.name) / "sample.wav"
        existing = self.voices / source.name
        sf.write(source, np.ones(2400) * 0.1, 24000)
        sf.write(existing, np.ones(2400) * 0.2, 24000)
        before = existing.read_bytes()
        with patch(
            "app.QFileDialog.getOpenFileName", return_value=(str(source), "WAV")
        ):
            self.window.add_voice()
        imported = self.voices / "sample (2).wav"
        self.assertEqual(existing.read_bytes(), before)
        self.assertEqual(imported.read_bytes(), source.read_bytes())
        self.assertEqual(self.window.library.currentData(), str(imported))


if __name__ == "__main__":
    unittest.main()
