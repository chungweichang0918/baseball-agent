"use strict";

async function connectAnalysis() {
  const status = document.getElementById("status");
  const title = status.querySelector("h3");
  const message = status.querySelector("p");
  try {
    const response = await fetch("./config.json", { cache: "no-store" });
    if (!response.ok) throw new Error("config unavailable");
    const config = await response.json();
    if (typeof config.analysis_url !== "string") throw new Error("invalid configuration");
    if (!config.analysis_url.trim()) {
      title.textContent = "分析服務尚未接上";
      message.textContent = "網站頁面已就緒；待分析後端部署完成並設定服務網址後，即可在這裡上傳影片。";
      return;
    }
    const url = new URL(config.analysis_url);
    const local = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
    if (url.protocol !== "https:" && !(local && url.protocol === "http:")) {
      throw new Error("HTTPS required");
    }
    if (url.username || url.password) throw new Error("credentials not allowed in URL");
    const frame = document.getElementById("analysis-frame");
    const link = document.getElementById("open-app");
    frame.src = url.href;
    frame.hidden = false;
    link.href = url.href;
    link.hidden = false;
    document.getElementById("embed-help").hidden = false;
    status.hidden = true;
    // An iframe load event cannot verify backend health across origins.
    // Keep the full-page link available rather than claiming the service is online.
  } catch (_) {
    title.textContent = "分析服務設定需要確認";
    message.textContent = "請確認 config.json 內填入有效的 HTTPS 分析網址。後端網址不應包含帳密或 API 金鑰。";
  }
}

connectAnalysis();
