// Run with: node site/tests/fish_names.test.cjs
// Display-name regression checks; DOM doubles do not verify WebGL or layout.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const expected = { huaban: '泡泡', hongding: '樱桃', lanyin: '阴影' };
const ids = Object.keys(expected);

function element() {
  return {
    children: [], _text: '', innerHTML: '', value: '', dataset: {}, attributes: {},
    clientWidth: 640, listeners: {}, classList: { toggle() {} },
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); },
    set textContent(value) { this._text = String(value); this.children = []; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this._text = ''; this.children = children; },
    setAttribute(key, value) { this.attributes[key] = String(value); },
    addEventListener(type, handler) { this.listeners[type] = handler; },
  };
}
async function checkTimeline(file, dataset, hours) {
  const html = read(file), data = JSON.parse(read(dataset)), before = JSON.stringify(data);
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, element()]));
  const get = id => { assert(nodes.has(id), `Missing element: ${id}`); return nodes.get(id); };
  if (nodes.has('metric-scope')) get('metric-scope').value = 'overall';
  if (nodes.has('metrics')) get('metrics').tBodies = [element()];
  const context = vm.createContext({
    document: { getElementById: get, createElement: element, querySelectorAll: () => [] },
    window: { addEventListener() {} }, location: { origin: 'http://localhost' },
    FMChart: { lines() {} }, setInterval, clearInterval,
    fetch: async () => ({ ok: true, json: async () => data }),
    console: { error(...args) { throw Error(args.join(' ')); } },
  });
  const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
  vm.runInContext(source, context, { filename: file });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(get('controls').attributes['aria-busy'], 'false');
  for (let i = 0; i < ids.length; i++) {
    assert.equal(data.fish[i].id, ids[i]);
    get('fish-tabs').listeners.click({ target: { closest: () => ({ dataset: { fish: String(i) } }) } });
    for (const hour of [0, Math.floor(hours / 2), hours - 1]) {
      get('timeline').listeners.input({ target: { value: String(hour) } });
      const labels = get('fish-tabs').children.map(button => button.textContent.split('H ')[0]);
      assert.deepEqual(labels, ids.map(id => expected[id]));
      assert(get('readout-title').textContent.startsWith(expected[ids[i]] + ' · '));
      assert.equal(get('readout-prob').textContent, `${(data.fish[i].points[hour].models.logistic.H * 100).toFixed(1)}%`);
    }
    if (nodes.has('metric-scope')) {
      get('metric-scope').value = ids[i];
      get('metric-scope').listeners.change();
      assert(get('metric-note').textContent.startsWith(expected[ids[i]] + '，'));
      get('timeline').listeners.input({ target: { value: '12' } });
      assert(get('day-title').textContent.endsWith(expected[ids[i]]));
    }
  }
  assert.equal(JSON.stringify(data), before, 'Display aliases must not mutate fetched source data');
  assert.equal(data.domain, 'synthetic');
}

(async () => {
  const home = read('site/index.html');
  assert.deepEqual([...home.matchAll(/<h3><span lang="en">(\w+)<\/span><span lang="zh-CN">(阴影|樱桃|泡泡)<\/span><\/h3>/g)].map(match => [match[1], match[2]]), [['Shadow', '阴影'], ['Cherry', '樱桃'], ['Bubbles', '泡泡']]);
  assert(/const shownNames=ZH\?names:\['Bubbles','Cherry','Shadow'\]/.test(read('webui/dist/app.js')));
  const app = read('webui/dist/app.js');
  const sceneNames = vm.runInNewContext(app.match(/const names=(\[[^\]]+\])/)[1]);
  assert.deepEqual(Array.from(sceneNames), ids.map(id => expected[id]));
  const companion = read('site/companion.html');
  const fish = vm.runInNewContext('(' + companion.match(/const FISH = (\{[\s\S]*?\n  \});/)[1] + ')');
  assert.deepEqual(ids.map(id => fish[id].name), ids.map(id => expected[id]));
  assert.deepEqual(ids.map(id => fish[id].session), ['test_08', 'test_02', 'test_05']);
  assert(read('site/models.html').includes('泡泡、樱桃和阴影'));
  for (const file of ['site/index.html', 'site/models.html', 'site/companion.html', 'site/ten-days.html', 'site/thirty-days.html']) {
    const text = read(file).replace(/黑白花斑|红顶浅色|蓝银条纹/g, '');
    assert(!/花斑|红顶|蓝银|蓝线|小珊瑚|红尾/.test(text), `Stale display name in ${file}`);
  }
  await checkTimeline('site/ten-days.html', 'webui/dist/assets/ten-days.json', 240);
  await checkTimeline('site/thirty-days.html', 'webui/dist/assets/thirty-days.json', 720);
  console.log('PASS: consistent names across home, introduction, 3D, companion and both timelines; IDs, probabilities, sessions and source data preserved.');
})().catch(error => { console.error(error); process.exitCode = 1; });
