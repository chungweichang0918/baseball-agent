# 接到 cwchang.me 的部署準備

## 已確認的網站線索

2026-10-30 公開查詢：cwchang.me 的 A 記錄為 GitHub Pages 的四個 185.199.*.153 位址，HTTP 首頁回應 `Server: GitHub.com`。據此判定目前網站使用 GitHub Pages。原站程式庫及可編輯權限仍待確認。

`github-pages/baseball/` 是可複製到原站的完整靜態頁目錄。發布後預計由 `/baseball/` 開啟；若原站使用 Jekyll 或其他建置工具，需確認該目錄會進入實際輸出。
在該目錄的 `config.json` 填入正式分析後端 HTTPS 網址。空白時會顯示「分析服務尚未接上」，不會假裝影片已可分析。
可在首頁新增 `<a href="/baseball/">棒球影片分析</a>`。這些檔案尚未提交到原站或發布。

## 獨立後端容器

專案根目錄已提供 `Dockerfile`。在有 Docker 的 Linux 主機，以專案根目錄作為建置目錄：

```sh
docker build -t diamond-studio .
docker run --name diamond-studio --restart unless-stopped \
  -p 127.0.0.1:7860:7860 --env-file /etc/bbagent.env \
  -v diamond-runs:/app/runs -v diamond-cache:/app/cache diamond-studio
```

環境檔可只放 `GEMINI_API_KEY=...`；沒有金鑰時可省略 `--env-file`，使用基礎分析流程。
映像使用 CPU 版 PyTorch，建置時下載 config.yaml 指定的官方 YOLO 模型。Docker context 排除本機影片、執行產物與 .env，且 Dockerfile 只複製指定來源。
使用託管容器平台時，對外服務埠設定為 7860，由平台提供 HTTPS；自行管理 VPS 則接下方 Nginx 範本。實際磁碟持久化方式依平台調整。
此環境未安裝 Docker，因此尚未執行映像建置或 Linux 容器測試。

候選平台仍需確認帳號與費用。Hugging Face 官方目前說明：建立 Gradio/Docker Space 需要付費方案；CPU Basic 雖沒有額外每小時費用，不能將整個部署稱為完全免費。
參考：https://huggingface.co/docs/hub/spaces-overview 。未購買或建立任何雲端資源。

這些是待套用的部署範本，目前沒有變更 DNS、原網站或主機設定，也沒有建立線上服務。

## 先確認主機

Namecheap 的網域管理畫面只表示網域註冊，不代表購買了可跑模型的主機。
請從 Hosting List 確認方案，並確認目前網站使用的平台。

- 只有網域：需要一台可以長期執行 Python / Gradio / PyTorch 的分析主機。
- Shared Hosting / cPanel：Namecheap 官方說明不支援 ASGI。原網站可繼續運作，分析後端部署在另外的 VPS 或相容平台，再由網站嵌入。
- VPS / Dedicated：可以評估在既有主機新增服務；需先確認 CPU、RAM、磁碟、作業系統與現有網站的資源使用。

官方來源：
- https://www.namecheap.com/support/knowledgebase/article.aspx/10048/2182/how-to-work-with-python-app/
- https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
- https://www.gradio.app/guides/sharing-your-app

## 建議的網址配置

原網站保留 `https://cwchang.me`，分析後端使用 **擬定的** `https://baseball.cwchang.me`。
先確認這個子網域沒有其他用途。部署完成後，在原網站加入連結或 `embed-snippet.html` 的 iframe。
網站若是 WordPress，可放進「自訂 HTML」區塊；iframe 的父網站與分析後端都必須使用有效 HTTPS。

子網域只需設定 `baseball` 記錄指向實際分析主機。只有確認 DNS 管理位置和主機 IP 後才能填入；不要替換根網域、www、MX 或整組 nameservers。

## 分析主機準備

1. 使用支援 Python 3.11–3.13 的 Linux 環境，安裝影片解碼所需的系統函式庫與 FFmpeg。
2. 將專案放在 `/opt/baseball-tactics-agent`，建立 `bbagent` 服務帳號並給予該專案的必要寫入權限；部署路徑不同時同步修改 service。
3. 建立 `.venv` 並安裝專案相依套件；模型放進 `models/`，或執行 `scripts/download_models.py`。不可把 Windows 的 `.venv` 複製到 Linux。
4. 如需 Gemini，把 `GEMINI_API_KEY` 放在伺服器的環境變數或 `/etc/bbagent.env`，限制檔案讀取權限。不要放在 HTML 或網站公開目錄。
5. 先在主機本機執行 `.venv/bin/python scripts/app.py --host 127.0.0.1 --port 7860`，驗證上傳與報告。
6. 依實際路徑調整 `bbagent.service`，作為長期執行服務。`cache/`、`runs/` 與 Gradio 暫存目錄需要可寫入。
7. 採子網域時用 `nginx-subdomain.conf`；這份只有 HTTP 反向代理，正式使用前必須完成該主機的 HTTPS 憑證與 HTTP→HTTPS 轉址。
8. 確認公開網址可以播放影片、上傳、接收串流狀態並下載報告後，再加入原網站連結或 iframe。

如果既有網站和分析服務在同一台可管理 Nginx 的主機，也可用 `/baseball/`。
把 `nginx-subpath.conf` 的 location 合併到既有 HTTPS server block，並在服務啟動命令加上 `--root-path /baseball`。
兩份 Nginx 範本是替代方案，勿直接覆蓋既有網站設定。

## 上線前實際驗證

目前保留單一模型分析併發，影片上傳上限預設 200 MB，可用 `--max-file-size` 調整。
Nginx 的 `client_max_body_size` 必須一起調整。分析時間也受 CPU 與片段長度影響。

現有程式把分析產物保存於 runs/，不應當成有帳號隔離的私人影片系統。
對外開放前須確認使用者範圍與影片保存期限，再決定登入、清理與容量政策。
不要把整個專案放到 public_html，或直接開放 Gradio 的 7860 埠給外網。

尚未驗證：目標主機安裝、Nginx 設定語法、實際 DNS、HTTPS、嵌入頁和目標主機效能。
