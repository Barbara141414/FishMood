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


def test_recorded_video_workflow_is_not_presented_as_live():
    home = (ROOT / 'index.html').read_text()
    models = (ROOT / 'models.html').read_text()
    companion = (ROOT / 'companion.html').read_text()
    for page in (home, models, companion):
        assert '放学后上传' in page
    for page in (home, companion):
        assert '直播' in page and ('开发' in page or '尚未接入' in page)
    assert 'clip_1200s.mp4' in home and '真实录像 · 非直播' in home
    assert '不是实时直播' in home
    assert '模拟实验' in models
    assert '两者不是同一份数据' in companion


def test_homepage_goes_from_video_to_named_fish_with_deployable_assets():
    home = (ROOT / 'index.html').read_text()
    assert '../outputs' not in home
    assert home.index('clip_1200s.mp4') < home.index('id="fish-title"')
    names = ['阴影', '樱桃', '泡泡']
    positions = [home.index(f'<span lang="zh-CN">{name}</span></h3>') for name in names]
    assert positions == sorted(positions)
    for photo in ('fish-yinying.jpg', 'fish-yingtao.jpg', 'fish-paopao.jpg'):
        assert f'assets/real/{photo}' in home and (ROOT / 'assets/real' / photo).is_file()
    assert '下一阶段' not in home and '只限这 8 秒' not in home


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


def test_display_names_keep_ids_and_domain_boundaries():
    result = run_core("""
      JSON.stringify(['huaban','hongding','lanyin'].map(function(id){
        return {
          real:window.__core.getSnapshot(id,{domain:'real'}),
          simulated:window.__core.getSnapshot(id,{domain:'synthetic'}),
          hello:window.__core.reply(id,'你好',{domain:'synthetic'})
        };
      }));
    """)
    for item, (fish_id, name) in zip(result, [
        ('huaban', '泡泡'), ('hongding', '樱桃'), ('lanyin', '阴影')
    ]):
        assert item['real']['fish_id'] == item['simulated']['fish_id'] == fish_id
        assert item['real']['name'] == item['simulated']['name'] == name
        assert item['real']['domain'] == 'real'
        assert item['simulated']['domain'] == 'synthetic'
        assert item['real']['identity_bound'] is False
        assert f'我是{name}' in item['hello']['text']
