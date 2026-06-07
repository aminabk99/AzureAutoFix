/**
 * AzureAutoFix — Content Script
 * Watches the DOM for Azure AD error codes and notifies the background worker.
 */

const AADSTS_PATTERN = /AADSTS\d{5,6}/g;

let lastDetectedCode = null;
let notificationShown = false;

function extractErrorFromPage() {
  const bodyText = document.body?.innerText || "";
  const titleText = document.title || "";
  const urlText = window.location.href;

  const sources = [bodyText, titleText, urlText].join(" ");
  const matches = sources.match(AADSTS_PATTERN);

  if (matches && matches.length > 0) {
    return matches[0]; // Return first match
  }
  return null;
}

function injectToast(errorCode, analysisResult) {
  // Remove existing toast if any
  const existing = document.getElementById("azureautofix-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.id = "azureautofix-toast";
  toast.innerHTML = `
    <div style="
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 999999;
      background: white;
      border: 1px solid #e0e0e0;
      border-left: 4px solid #0078d4;
      border-radius: 8px;
      padding: 16px 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      font-family: 'Segoe UI', sans-serif;
      font-size: 14px;
      max-width: 320px;
      animation: slideIn 0.3s ease;
    ">
      <div style="display:flex; align-items:center; margin-bottom:8px;">
        <span style="font-size:18px; margin-right:8px;">⚡</span>
        <strong style="color:#0078d4;">AzureAutoFix</strong>
        <span id="azureautofix-close" style="margin-left:auto; cursor:pointer; color:#666; font-size:16px;">✕</span>
      </div>
      <div style="color:#333; margin-bottom:4px;">
        Error detected: <code style="background:#f0f7ff; padding:2px 6px; border-radius:4px;">${errorCode}</code>
      </div>
      <div style="color:#666; font-size:12px; margin-bottom:12px;">
        ${analysisResult?.user_message || "Analyzing..."}
      </div>
      <div style="display:flex; gap:8px;">
        <button id="azureautofix-fix-btn" style="
          background:#0078d4; color:white; border:none; border-radius:6px;
          padding:6px 14px; font-size:13px; cursor:pointer; font-weight:600;
        ">Fix automatically</button>
        <button id="azureautofix-details-btn" style="
          background:white; color:#0078d4; border:1px solid #0078d4; border-radius:6px;
          padding:6px 14px; font-size:13px; cursor:pointer;
        ">Details</button>
      </div>
    </div>
    <style>
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to   { transform: translateX(0);   opacity: 1; }
      }
    </style>
  `;

  document.body.appendChild(toast);

  document.getElementById("azureautofix-close").onclick = () => toast.remove();

  document.getElementById("azureautofix-fix-btn").onclick = () => {
    chrome.runtime.sendMessage({
      type: "FIX_ERROR",
      errorCode,
      analysisResult,
    });
    toast.querySelector("div > div:nth-child(3)").textContent = "Fixing... check the extension popup for progress.";
  };

  document.getElementById("azureautofix-details-btn").onclick = () => {
    chrome.runtime.sendMessage({ type: "OPEN_POPUP", errorCode, analysisResult });
  };
}

// ── Main detection loop ─────────────────────────────────────────────────────

function checkForErrors() {
  const code = extractErrorFromPage();
  if (!code || code === lastDetectedCode) return;

  lastDetectedCode = code;
  console.log(`[AzureAutoFix] Detected: ${code}`);

  chrome.runtime.sendMessage({ type: "ERROR_DETECTED", errorCode: code }, (response) => {
    if (response?.analysisResult) {
      injectToast(code, response.analysisResult);
    } else {
      injectToast(code, { user_message: "Analyzing error..." });
    }
  });
}

// Watch on load and on DOM changes
checkForErrors();

const observer = new MutationObserver(() => checkForErrors());
observer.observe(document.body, { childList: true, subtree: true, characterData: true });
