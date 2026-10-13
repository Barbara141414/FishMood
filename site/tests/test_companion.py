"""Guard the real/synthetic boundary in the static companion demo.

The macOS JavaScriptCore command is used because this workspace does not require Node.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_core(expression):
    if shutil.which("osascript") is None:
        pytest.skip("macOS JavaScriptCore is unavailable")
    page = (ROOT / "companion.html").read_text()
    script = page.split("<script>\n(() => {", 1)[1].split("  // ---------- Interface ----------", 1)[0]
    data = (ROOT / "assets/data.js").read_text()
    source = (
        "var window=this;\n" + data + "\n(() => {" + script +
        "window.__core={getSnapshot,reply};})();\n" + expression
    )
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", source],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_real_snapshot_is_unknown_even_after_simulation_is_read():
    result = run_core("""
      var sim=window.__core.getSnapshot('huaban',{domain:'synthetic'});
      var real=window.__core.getSnapshot('huaban',{domain:'real'});
      JSON.stringify({sim:sim,real:real});
    """)
    sim, real = result["sim"], result["real"]
    assert sim["domain"] == "synthetic"
    assert sim["visible"] and sim["visible"][0]["source"] == "sim_track"
    assert all(item["source"].startswith("synthetic:model:") for item in sim["estimate"].values())
    assert real["domain"] == "real"
    assert real["tank"]["fish_count"] == 3
    assert real["identity_bound"] is False
    assert real["updated_at"] is None
    assert real["recorded_at"] is None
    assert real["tank"]["recording"] == "2026-10-04 10:20–11:39"
    assert real["visible"] == []
    assert all(item == {"label": "unknown", "p": None, "source": None} for item in real["estimate"].values())


def test_real_chat_refuses_and_simulated_chat_labels_limits():
    result = run_core("""
      JSON.stringify({
        real:window.__core.reply('hongding','今天吃饱了吗？',{domain:'real'}),
        food:window.__core.reply('hongding','今天吃饱了吗？',{domain:'synthetic'}),
        health:window.__core.reply('hongding','你是不是生病了？',{domain:'synthetic'}),
        hello:window.__core.reply('hongding','你好呀',{domain:'synthetic'})
      });
    """)
    assert result["real"]["domain"] == "real"
    assert "不能替画面里的鱼回答" in result["real"]["text"]
    assert "synthetic" not in result["real"]["text"]
    assert result["food"]["domain"] == "synthetic"
    assert "不能确认真实鱼的进食" in result["food"]["text"]
    assert any(tag[0] == "ref" for tag in result["food"]["tags"])
    assert any(tag[0] == "ref" for tag in result["health"]["tags"])
    assert any(tag[0] == "per" for tag in result["hello"]["tags"])
    assert result["hello"]["snapshot_at"].startswith("模拟 t=")
