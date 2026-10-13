// 唯一需要手动填写的地方：仓库地址。留空时页面显示“GitHub 待填”。
window.FISHMOOD_CONFIG = {
  githubUrl: 'https://github.com/Barbara141414/FishMood',
  // 3D 模拟回放（已有）：用 `python3 -m http.server 8080` 从项目根目录启动时可访问
  simDemoUrl: '../webui/dist/index.html',
};

document.addEventListener('DOMContentLoaded', () => {
  const url = window.FISHMOOD_CONFIG.githubUrl;
  document.querySelectorAll('[data-gh]').forEach((a) => {
    if (url) { a.href = url; a.classList.remove('todo'); a.querySelector('span').textContent = 'GitHub'; }
    else { a.removeAttribute('href'); a.classList.add('todo'); a.querySelector('span').textContent = 'GitHub 待填'; a.title = '在 site/assets/config.js 填入 githubUrl'; }
  });
});
