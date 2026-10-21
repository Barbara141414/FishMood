# FishMood · 我的鱼儿朋友

记录我鱼缸里三条神仙鱼（阴影、樱桃、泡泡）的日常，并尝试用编程和模型把观察整理得更清楚。

在线访问：<https://barbara141414.github.io/FishMood/>

- **看鱼**（`site/index.html`）：鱼缸录像与三条鱼的卡片。
- **项目介绍**（`site/models.html`）：三维鱼缸、实拍逐鱼标注，以及逻辑回归与隐马尔可夫模型的实验结果。

录像来自我每天放学后上传的当天视频，不是直播；实时直播接入还在开发。

## 本地预览

在仓库根目录运行：

```bash
python3 -m http.server 8080
```

然后打开 <http://127.0.0.1:8080/>。

## 目录

- `site/`：网页与素材
- `webui/dist/`：三维鱼缸（Three.js）
- `.github/workflows/pages.yml`：推送到 `main` 后自动发布到 GitHub Pages
