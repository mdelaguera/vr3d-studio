"""Non-UI logic behind the app: preferences, recent files, setup diagnosis."""

import json

import pytest

from gui import prefs, system_check


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "any2vr.json"
    monkeypatch.setattr(prefs, "SETTINGS_PATH", str(path))
    monkeypatch.setattr(prefs, "LEGACY_SETTINGS_PATH", str(tmp_path / "legacy.json"))
    return path


def test_load_drops_stale_values_and_missing_recents(settings_file, tmp_path):
    real = tmp_path / "clip_180_SBS.mp4"
    real.write_bytes(b"x")
    settings_file.write_text(json.dumps({
        "model": "vitl",                      # removed non-commercial model
        "mode": "vr180",
        "recent": [{"path": str(real), "kind": "video"}, {"path": str(tmp_path / "gone.mp4"), "kind": "video"}, "junk"],
        "unknown_key": 1,
    }))
    s = prefs.load()
    assert s["model"] == prefs.DEFAULTS["model"] and s["mode"] == "vr180"
    assert [r["path"] for r in s["recent"]] == [str(real)]
    assert "unknown_key" not in s


def test_legacy_settings_are_read_when_new_file_missing(settings_file, tmp_path):
    (tmp_path / "legacy.json").write_text(json.dumps({"mode": "anaglyph"}))
    assert prefs.load()["mode"] == "anaglyph"


def test_corrupt_settings_fall_back_to_defaults(settings_file):
    settings_file.write_text("{not json")
    assert prefs.load()["model"] == prefs.DEFAULTS["model"]


def test_add_recent_dedupes_and_caps():
    recent = []
    for i in range(12):
        recent = prefs.add_recent(recent, f"/f{i}.mp4", "video")
    recent = prefs.add_recent(recent, "/f5.mp4", "video")
    assert len(recent) == prefs.MAX_RECENT
    assert recent[0]["path"] == "/f5.mp4" and [r["path"] for r in recent].count("/f5.mp4") == 1


def test_save_roundtrip(settings_file):
    s = prefs.load()
    s["welcomed"] = True
    prefs.save(s)
    assert prefs.load()["welcomed"] is True


def _fake_env(monkeypatch, device, adapters, ffmpeg=True, encoder=""):
    import engine.depth as depth
    import engine.media as media
    monkeypatch.setattr(depth, "pick_device", lambda: device)
    monkeypatch.setattr(depth, "device_label", lambda: "CPU (no GPU acceleration)" if device == "cpu" else "RTX · CUDA")
    monkeypatch.setattr(media, "hardware_encoder", lambda: encoder)
    monkeypatch.setattr(system_check, "_display_adapters", lambda: adapters)
    monkeypatch.setattr(system_check.shutil, "which", lambda exe: "/bin/ffmpeg" if ffmpeg else None)


def test_nvidia_gpu_with_cpu_only_pytorch_gets_cuda_fix(monkeypatch):
    _fake_env(monkeypatch, "cpu", ["NVIDIA GeForce RTX 5070"], encoder="hevc_nvenc")
    r = system_check.check()
    assert not r.has_gpu and len(r.issues) == 1 and r.encoder == "NVIDIA GPU"
    assert "RTX 5070" in r.issues[0].title and "cu128" in r.issues[0].fix_command


def test_amd_gpu_is_named_and_gets_no_cuda_advice(monkeypatch):
    _fake_env(monkeypatch, "cpu", ["Meta Virtual Monitor", "AMD Radeon RX 6800 XT"], encoder="hevc_amf")
    r = system_check.check()
    issue = r.issues[0]
    assert "AMD Radeon RX 6800 XT" in issue.title and not issue.fix_command
    assert r.encoder == "AMD GPU"


def test_cpu_only_machine_gets_advice_without_a_fix_command(monkeypatch):
    _fake_env(monkeypatch, "cpu", [])
    r = system_check.check()
    assert r.issues[0].title == "Running on CPU" and not r.issues[0].fix_command and r.encoder == "CPU"


def test_missing_ffmpeg_is_blocking(monkeypatch):
    _fake_env(monkeypatch, "cuda", [], ffmpeg=False)
    r = system_check.check()
    assert r.has_gpu and [i.blocking for i in r.issues] == [True]


def test_virtual_adapters_are_ignored(monkeypatch):
    monkeypatch.setattr(system_check.sys, "platform", "win32")
    monkeypatch.setattr(system_check.subprocess, "run", lambda *a, **k: type("R", (), {
        "stdout": "Meta Virtual Monitor\nMicrosoft Basic Display Adapter\nAMD Radeon RX 6800 XT\n"})())
    assert system_check._display_adapters() == ["AMD Radeon RX 6800 XT"]
