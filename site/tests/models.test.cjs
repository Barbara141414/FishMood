// Run with: node site/tests/models.test.cjs
// DOM doubles validate rendering logic, not browser layout or WebGL.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');

function dom(html = '', width = 640) {
  const nodes = new Map();
  function element(tag = 'div') {
    return {
      tag, children: [], attributes: {}, innerHTML: '', textContent: '', value: '',
      clientWidth: width, dataset: {}, classList: { toggle() {} },
      style: { setProperty(key, value) { this[key] = value; } },
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; },
      setAttribute(key, value) { this.attributes[key] = value; },
      querySelectorAll() { return []; },
    };
  }
  for (const [, id] of html.matchAll(/id="([^"]+)"/g)) nodes.set(id, element());
  return {
    nodes, createElement: element,
    getElementById(id) {
      assert(nodes.has(id), `Missing DOM target: ${id}`);
      return nodes.get(id);
    },
  };
}
function pick(document, id, value) {
  const button = { dataset: { v: value } };
  document.getElementById(id).onclick({ target: { closest: () => button } });
}

function testIntroduction(width) {
  const html = read('site/models.html');
  assert(!/GAM|三种读法|三个模型/.test(html));
  assert(html.includes('index.html?models=logistic,hmm'));
  assert(html.includes('repeat(2, minmax(0, 1fr))'));
  assert(!/三分钟讲稿|Algorithm Modeling|先把“读心”/.test(html));
  assert(html.includes('想更好地关心，<br>我的鱼儿朋友。'));
  assert(!html.includes('当前可体验') && !html.includes('真实数据接入计划') && !html.includes('id="replay"') && !html.includes('id="ambiguity"'));
  assert(!html.includes('id="data-source"') && !html.includes('href="#data-source"'));
  const detailIds = ['logisticDetails', 'hmmDetails', 'resultDetails'];
  for (const id of detailIds) {
    const tag = html.match(new RegExp(`<details[^>]*id="${id}"[^>]*>`));
    assert(tag, `Missing project evidence disclosure: ${id}`);
    assert(!/\sopen(?:[\s=>])/.test(tag[0]), 'Technical details must start collapsed');
  }
  const home = read('site/index.html');
  assert(html.includes('FishMood · 项目介绍') && home.includes('FISHMOOD · 我的鱼缸观察记录'));
  assert(home.includes('<h1>看看<br><em>我的鱼</em></h1>'));
  assert(!home.includes('静态界面样例'));
  assert(html.includes('assets/real/track-0500.m4v') && !html.includes('../outputs'));
  assert(!/此刻的三个小朋友|活跃、稳定与探索|视频为公开鱼缸录像/.test(home));
  const document = dom(html, width);
  const window = { addEventListener() {} };
  const context = vm.createContext({ document, window, setTimeout, clearTimeout });
  for (const file of ['site/assets/data.js', 'site/assets/charts.js']) {
    vm.runInContext(read(file), context, { filename: file });
  }
  const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script, context, { filename: 'models.html:inline' });
  const get = id => document.getElementById(id);
  assert.equal((get('metricTable').innerHTML.match(/<tr>/g) || []).length, 7);
  assert.equal((get('metricTable').innerHTML.match(/rowspan="2"/g) || []).length, 3);
  assert.equal((get('scatter').innerHTML.match(/<circle /g) || []).length, 6);
  assert(!get('scatterLegend').innerHTML.includes('GAM'));
  for (const id of detailIds) {
    const detail = get(id);
    get('scatter').innerHTML = 'closed chart';
    detail.open = false;
    detail.ontoggle();
    assert.equal(get('scatter').innerHTML, 'closed chart');
    for (const chart of ['scatter', 'coefChart', 'hmmGap']) get(chart).clientWidth = width + 40;
    detail.open = true;
    detail.ontoggle();
    for (const chart of ['scatter', 'coefChart', 'hmmGap']) {
      assert(get(chart).innerHTML.includes(`viewBox="0 0 ${width + 40} `), `${chart} must redraw at the revealed width`);
      get(chart).clientWidth = width;
    }
  }
  for (const dim of ['H', 'S', 'E']) {
    pick(document, 'dimTabs', dim);
    assert(get('coefChart').innerHTML.includes('<svg'));
    assert.equal((get('hmmMatrix').innerHTML.match(/class="mono"/g) || []).length, 4);
    assert(get('hmmGap').innerHTML.includes('<svg'));
  }
}

async function testReplay(search, expected) {
  const source = read('webui/dist/replay.js');
  const data = JSON.parse(read('webui/dist/assets/model-replay.json'));
  const original = JSON.stringify(data);
  const document = dom(read('webui/dist/index.html') + source);
  const get = id => document.getElementById(id);
  get('replay-state').value = 'hunger';
  get('replay-window').value = '0';
  const context = vm.createContext({
    document, location: { search }, URLSearchParams,
    fetch: async () => ({ ok: true, json: async () => data }),
    console: { error(...args) { throw Error(args.join(' ')); } },
  });
  await vm.runInContext(`(async () => {${source}\n})()`, context);
  assert.equal(get('replay-content').attributes['aria-busy'], 'false');
  for (const session of data.sessions) {
    get('replay-session').value = session.id;
    get('replay-session').onchange();
    for (const state of ['hunger', 'stress', 'exploration', 'fatigue']) {
      get('replay-state').value = state;
      get('replay-state').onchange();
      for (let index = 0; index < session.windows.length; index++) {
        get('replay-window').value = String(index);
        get('replay-window').oninput();
        const cards = get('replay-models').children;
        assert.equal(cards.length, expected.length);
        assert.equal(get('replay-models').style['--replay-model-count'], String(expected.length));
        cards.forEach((card, i) => {
          const backend = expected[i];
          const prediction = session.windows[index].predictions[backend]['sim_' + state];
          assert.equal(card.children[0].textContent.split(' ')[0], { logistic: 'LR', gam: 'GAM', hmm: 'HMM' }[backend]);
          const probability = card.children.find(child => child.className === 'probability-value');
          const high = prediction.probabilities?.high;
          assert.equal(probability.textContent, Number.isFinite(high) ? `${(high * 100).toFixed(1)}%` : '—');
        });
      }
    }
  }
  assert.equal(JSON.stringify(data), original, 'Display filtering must not mutate the dataset');
}

(async () => {
  for (const width of [244, 640]) testIntroduction(width);
  await testReplay('?models=logistic,hmm', ['logistic', 'hmm']);
  await testReplay('', ['logistic', 'gam', 'hmm']);
  await testReplay('?models=unknown', ['logistic', 'gam', 'hmm']);
  await testReplay('?models=hmm,logistic,logistic,unknown', ['logistic', 'hmm']);
  console.log('PASS: two-model introduction, 3 dimensions at 2 chart widths; replay filters/default/fallback over all 16 × 4 × 12 windows; missing probabilities preserved.');
})().catch(error => { console.error(error); process.exitCode = 1; });
