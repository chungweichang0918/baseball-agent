# 使用自己的電腦展示

已選定的來源專案： https://github.com/chungweichang0918/baseball-agent 。
2026-10-30 查詢時遠端是空的、尚未啟用 Pages；它不是目前 cwchang.me 個人首頁的既有來源。

## 首次發布

1. 使用既有 Git Credential Manager 登入，或在終端執行 `gh auth login`。本機已驗證 Git Credential Manager 可推送到此專案。不要把 token 貼進聊天。
2. 把本專案來源提交到你提供的 baseball-agent 專案（排除 .env、.venv、runs、cache、影片與模型權重，已有 .gitignore）。遠端已有內容時先同步，不要強制推送。
3. 到該專案 Settings → Pages，Build and deployment 的 Source 選 GitHub Actions。然後執行 Actions 裡的 Publish baseball demo website，或推送 docs/ 變更觸發部署。
4. 初始網址為 https://chungweichang0918.github.io/baseball-agent/baseball/ 。這是預計網址，Actions 成功後才會存在。

Pages 只上傳 docs/ 靜態目錄。Python、API 金鑰與本機影片不會被放進 Pages 產物。GitHub 專案若設成公開，已提交的其他程式碼也會公開，與 Pages 產物是不同層次。

## 每次展示

Windows 雙擊根目錄的 `start-demo.cmd`，或在專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe -u scripts\demo.py --publish
```

程式在本機執行模型，以 Gradio 分享連結提供臨時 HTTPS 服務，將連結更新到 docs/baseball/config.json。
`--publish` 使用 GitHub CLI 或既有 Git Credential Manager 更新遠端同一個設定檔；Pages 工作流程隨後重新發布，所以啟動後需等待部署完成。
如果尚未登入或來源尚未首次發布，會保留直接展示連結，並清楚告知網站設定尚未更新。

展示期間保持電腦、視窗與網路開啟；取得連結的人可上傳影片使用此分析服務。
正常按 Ctrl+C 結束時會清空本機和已發布的連線設定。若有更新的另一場展示，舊場次不會清除新連結。
強制關機／關閉視窗不保證能完成清理，下一次啟動會更新連結。分享連結有有效期，不能當永久雲端服務。
原站跨來源嵌入若遇到瀏覽器限制，可用頁面上的「以完整頁面開啟」。

## 接到 Namecheap 網域

可以先從原個人首頁連到上述 Pages 網址；修改原首頁需要它真正的來源專案權限。
也可把這個獨立展示站設為 baseball.cwchang.me：

1. 在 baseball-agent 的 Pages 設定，Custom domain 設為 baseball.cwchang.me。
2. 在 Namecheap 的 Advanced DNS 新增 CNAME：Host 填 baseball，Value 填 chungweichang0918.github.io。
3. 等待 GitHub 的 DNS 與憑證驗證成功，啟用 Enforce HTTPS。

只針對 baseball 子網域操作；根網域的現有 A 記錄保留。目前沒有執行這些遠端設定。
使用子網域後，站台根目錄會轉到 /baseball/；兩者都使用同一份 docs/baseball/config.json。

此方案不需要 Namecheap 主機或付費雲端後端。外部分享服務的可用性與有效期仍依 Gradio 服務為準。

## 此電腦目前的驗證結果

已成功推送來源、啟用 GitHub Pages 並發出部署要求。本機分析可啟動，但 Gradio 的 frpc 分享程式在 CreateProcess 階段被 Windows 拒絕執行（WinError 5），因此尚未取得公開分析連結。未修改防毒或系統安全設定。此限制須在電腦上釐清後，再執行 start-demo.cmd；取得成功連結前，網站會保留「分析服務尚未接上」。
