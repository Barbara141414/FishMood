// Tiny SVG chart helpers. No dependencies.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const FEATURE_ZH = {
    speed_mean: '平均速度', acceleration_mean: '平均加速度', displacement_ratio: '位移/路程比',
    turn_rate: '转向频率', feeding_zone_fraction: '喂食区停留', surface_fraction: '近水面停留',
    dispersion: '与鱼群距离', activity_fraction: '活跃时间占比', wall_fraction: '贴缸壁停留',
    shelter_fraction: '躲藏区停留', speed_baseline_z: '速度相对本鱼基线', feeding_baseline_z: '喂食区相对本鱼基线',
  };
  const FEATURE_EN = {
    speed_mean: 'Mean speed', acceleration_mean: 'Mean acceleration', displacement_ratio: 'Displacement / path',
    turn_rate: 'Turn rate', feeding_zone_fraction: 'Time in feeding area', surface_fraction: 'Time near surface',
    dispersion: 'Distance from group', activity_fraction: 'Active time share', wall_fraction: 'Time by the glass',
    shelter_fraction: 'Time in shelter', speed_baseline_z: 'Speed vs. own baseline', feeding_baseline_z: 'Feeding area vs. own baseline',
  };
  const DIM_ZH = { H: '摄食相关 H', S: '警觉/回避 S', E: '探索 E' };
  const DIM_EN = { H: 'Feeding H', S: 'Alert / avoidant S', E: 'Exploring E' };
  const MODEL_ZH = { logistic: 'Logistic', gam: 'GAM', hmm: 'HMM' };
  const COLORS = { logistic: '#1f5f5b', gam: '#c27d2a', hmm: '#667581', truth: '#1e2b2a', H: '#c27d2a', S: '#bf5a3f', E: '#2f7f78' };

  function lines(el, opt) {
    // opt: {series:[{name,color,points:[[x,y|null]],dash,width}], xDomain, yDomain, xLabel, yLabel, bands:[{x0,x1,color,label}], vlines:[{x,label}], hlines:[{y,label}], height, xTicks, yTicks, xFmt}
    const W = el.clientWidth || 640, H = opt.height || 260, m = { l: 44, r: 14, t: 16, b: 34 };
    const [x0, x1] = opt.xDomain, [y0, y1] = opt.yDomain;
    const sx = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
    const sy = (y) => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
    const fmt = opt.xFmt || ((v) => v);
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img">`;
    (opt.bands || []).forEach((b) => {
      s += `<rect x="${sx(b.x0)}" y="${m.t}" width="${Math.max(2, sx(b.x1) - sx(b.x0))}" height="${H - m.t - m.b}" fill="${b.color}" opacity=".18"/>`;
      if (b.label) s += `<text x="${sx(b.x1) + 4}" y="${m.t + 12 + (b.row || 0) * 14}" font-size="11" fill="${b.color}">${esc(b.label)}</text>`;
    });
    const yT = opt.yTicks || 5;
    for (let i = 0; i <= yT; i++) {
      const v = y0 + (y1 - y0) * i / yT, y = sy(v);
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="#d8d0c0" stroke-width="1"/>`;
      s += `<text x="${m.l - 6}" y="${y + 4}" font-size="11" text-anchor="end" fill="#7d8987">${opt.yFmt ? opt.yFmt(v) : +v.toFixed(2)}</text>`;
    }
    const xT = opt.xTicks || 6;
    for (let i = 0; i <= xT; i++) {
      const v = x0 + (x1 - x0) * i / xT;
      s += `<text x="${sx(v)}" y="${H - m.b + 16}" font-size="11" text-anchor="middle" fill="#7d8987">${fmt(v)}</text>`;
    }
    (opt.hlines || []).forEach((h) => {
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(h.y)}" y2="${sy(h.y)}" stroke="#7d8987" stroke-dasharray="4 4"/>`;
      if (h.label) s += `<text x="${W - m.r}" y="${sy(h.y) - 4}" font-size="11" text-anchor="end" fill="#7d8987">${esc(h.label)}</text>`;
    });
    (opt.vlines || []).forEach((v) => {
      s += `<line x1="${sx(v.x)}" x2="${sx(v.x)}" y1="${m.t}" y2="${H - m.b}" stroke="${v.color || '#7d8987'}" stroke-dasharray="3 3"/>`;
      if (v.label) s += `<text x="${sx(v.x) + 4}" y="${m.t + 11}" font-size="11" fill="${v.color || '#7d8987'}">${esc(v.label)}</text>`;
    });
    opt.series.forEach((se) => {
      let d = '', pen = false;
      se.points.forEach(([x, y]) => {
        if (y === null || y === undefined || Number.isNaN(y)) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(Math.max(y0, Math.min(y1, y))).toFixed(1)}`; pen = true;
      });
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ''} stroke-linejoin="round" stroke-linecap="round"/>`;
      if (se.dots) se.points.forEach(([x, y]) => { if (y !== null) s += `<circle cx="${sx(x)}" cy="${sy(y)}" r="3" fill="${se.color}"/>`; });
    });
    if (opt.xLabel) s += `<text x="${W - m.r}" y="${H - 4}" font-size="11" text-anchor="end" fill="#7d8987">${esc(opt.xLabel)}</text>`;
    if (opt.yLabel) s += `<text x="${m.l}" y="${m.t - 4}" font-size="11" fill="#7d8987">${esc(opt.yLabel)}</text>`;
    el.innerHTML = s + '</svg>';
  }

  function hbars(el, rows, opt = {}) {
    // rows: [[label, value]]
    const W = el.clientWidth || 420, rowH = 26, m = { l: 150, r: 46, t: 6 };
    const H = m.t * 2 + rows.length * rowH;
    const max = opt.max || Math.max(...rows.map((r) => Math.abs(r[1])), 1e-6);
    const cx = m.l + (W - m.l - m.r) / 2, half = (W - m.l - m.r) / 2;
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">`;
    s += `<line x1="${cx}" x2="${cx}" y1="0" y2="${H}" stroke="#7d8987"/>`;
    rows.forEach(([lab, v], i) => {
      const y = m.t + i * rowH, w = Math.abs(v) / max * half;
      const col = v >= 0 ? (opt.pos || '#1f5f5b') : (opt.neg || '#bf5a3f');
      s += `<text x="${m.l - 10}" y="${y + 17}" font-size="12.5" text-anchor="end" fill="#1e2b2a">${esc(lab)}</text>`;
      s += `<rect x="${v >= 0 ? cx : cx - w}" y="${y + 5}" width="${w}" height="16" rx="3" fill="${col}"/>`;
      s += `<text x="${v >= 0 ? cx + w + 5 : cx - w - 5}" y="${y + 17}" font-size="11" font-family="JetBrains Mono, monospace" text-anchor="${v >= 0 ? 'start' : 'end'}" fill="#4a5856">${v > 0 ? '+' : ''}${v.toFixed(2)}</text>`;
    });
    el.innerHTML = s + '</svg>';
  }

  window.FMChart = { lines, hbars, FEATURE_ZH, FEATURE_EN, DIM_ZH, DIM_EN, MODEL_ZH, COLORS, esc };
})();
