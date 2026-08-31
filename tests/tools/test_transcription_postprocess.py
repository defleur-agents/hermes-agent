from pathlib import Path

from tools import transcription_tools as tt


VOICE_MIC_RULES = {
    "normalize_spoken_times": True,
    "replacements": [
        {"pattern": r"\bNovera\b", "replacement": "Novaire"},
        {"pattern": r"\bSony\b(?=(?:['\u2019]s)\b)", "replacement": "Somi"},
        {
            "pattern": (
                r"\bM[\s.-]*[IO][\s.-]*[ZS](?:[\s.-]*[ZS])?[\s.-]*E[\s.-]*L"
                r"(?:[\s.-]*L)?(?:[\s.-]*E)?(?:\s*(?:\.\s*)?AI)?\b"
            ),
            "replacement": "Mizel.ai",
        },
    ],
}


def test_postprocess_applies_regex_rules_and_spoken_time_normalization():
    text = "Ask Novera about Sony's notes for M O Z E L AI at 09:00 p.m."

    assert tt.apply_transcription_postprocess(
        text, {"postprocess": VOICE_MIC_RULES}
    ) == "Ask Novaire about Somi's notes for Mizel.ai at 9pm"


def test_postprocess_skips_invalid_rules_without_losing_valid_ones():
    config = {
        "postprocess": {
            "replacements": [
                {"pattern": "[", "replacement": "broken"},
                {"pattern": r"\bNovera\b", "replacement": "Novaire"},
            ]
        }
    }

    assert tt.apply_transcription_postprocess("Novera", config) == "Novaire"


def test_transcribe_audio_postprocesses_successful_provider_result(monkeypatch, tmp_path):
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fake audio")

    monkeypatch.setattr(tt, "_validate_audio_source_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tt,
        "_prepare_audio_for_transcription",
        lambda path: (path, None, None),
    )
    monkeypatch.setattr(tt, "_validate_audio_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tt,
        "_transcribe_prepared_audio",
        lambda *args, **kwargs: {
            "success": True,
            "transcript": "Novera owns the title",
            "provider": "fake",
        },
    )
    monkeypatch.setattr(
        tt,
        "_load_stt_config",
        lambda: {"postprocess": VOICE_MIC_RULES},
    )

    result = tt.transcribe_audio(str(audio_path))

    assert result == {
        "success": True,
        "transcript": "Novaire owns the title",
        "provider": "fake",
    }


def test_postprocess_leaves_failed_results_untouched():
    result = {"success": False, "transcript": "Novera", "error": "failed"}

    assert tt._postprocess_transcription_result(
        result, {"postprocess": VOICE_MIC_RULES}
    ) == result
