// Shared EN / 中文 switch. English is the default; the choice is remembered.
// Markup: <span lang="en">…</span><span lang="zh-CN">…</span>; attributes via data-<attr>-en / data-<attr>-zh.
(function () {
  const KEY = 'fishmood-lang', ATTRS = ['alt', 'aria-label', 'title', 'content'];
  const root = document.documentElement;
  let lang = null;
  try { lang = new URLSearchParams(location.search).get('lang') || localStorage.getItem(KEY); } catch (e) { /* storage unavailable */ }
  lang = lang === 'zh' ? 'zh' : 'en';

  function apply() {
    root.dataset.lang = lang;
    root.lang = lang === 'zh' ? 'zh-CN' : 'en';
    for (const attr of ATTRS) {
      document.querySelectorAll(`[data-${attr}-en]`).forEach((el) => el.setAttribute(attr, el.getAttribute(`data-${attr}-${lang}`)));
    }
    document.querySelectorAll('title[data-en]').forEach((el) => { el.textContent = el.dataset[lang]; });
    document.querySelectorAll('iframe[data-follow-lang]').forEach((frame) => {
      const url = new URL(frame.getAttribute('src'), location.href);
      if (url.searchParams.get('lang') === lang) return;
      url.searchParams.set('lang', lang);
      frame.setAttribute('src', url.pathname.replace(/^.*\/webui\//, '../webui/') + url.search);
    });
  }

  function set(next) {
    lang = next === 'zh' ? 'zh' : 'en';
    try { localStorage.setItem(KEY, lang); } catch (e) { /* storage unavailable */ }
    apply();
    window.dispatchEvent(new CustomEvent('fishmood:lang', { detail: lang }));
  }

  root.dataset.lang = lang;
  root.lang = lang === 'zh' ? 'zh-CN' : 'en';
  window.FMLang = { get: () => lang, set, t: (en, zh) => (lang === 'zh' ? zh : en) };
  document.addEventListener('DOMContentLoaded', () => {
    apply();
    document.querySelectorAll('[data-lang-toggle]').forEach((btn) => { btn.onclick = () => set(lang === 'zh' ? 'en' : 'zh'); });
  });
})();
