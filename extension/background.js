/**
 * AzureAutoFix — Background Service Worker
 * Handles API calls to the FastAPI backend.
 * Content script messages come here; we call the API and respond.
 */

// Live backend (deployed on Railway). For local development, change this to
// "http://localhost:8000" and add it back to host_permissions in manifest.json.
const API_BASE = "https://azureautofix-production.up.railway.app";

// Public demo API key for the /fix endpoint. This only gates the endpoint
// itself — every /fix call still requires the caller's own MS Graph
// access_token, so this key cannot be used to modify anyone else's tenant.
const DEMO_API_KEY = "Ts3Rh5YOal1Ln_6WoJr05-eYaottLK05";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ERROR_DETECTED") {
    handleErrorDetected(message.errorCode).then(sendResponse);
    return true; // Keep channel open for async response
  }

  if (message.type === "FIX_ERROR") {
    handleFix(message.errorCode, message.analysisResult).then(sendResponse);
    return true;
  }

  if (message.type === "OPEN_POPUP") {
    // Store in chrome.storage for popup.js to read
    chrome.storage.local.set({
      lastError: message.errorCode,
      lastAnalysis: message.analysisResult,
    });
    chrome.action.openPopup();
    sendResponse({ ok: true });
  }
});

async function handleErrorDetected(errorCode) {
  try {
    const resp = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error_input: errorCode }),
    });
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    const analysisResult = await resp.json();

    // Store for popup
    await chrome.storage.local.set({ lastError: errorCode, lastAnalysis: analysisResult });

    // Badge on extension icon
    chrome.action.setBadgeText({ text: "!" });
    chrome.action.setBadgeBackgroundColor({ color: "#d13438" });

    return { analysisResult };
  } catch (err) {
    console.error("[AzureAutoFix] Analysis failed:", err);
    return { error: err.message };
  }
}

async function handleFix(errorCode, analysisResult) {
  const { accessToken, userId, appId, redirectUri } = await chrome.storage.local.get([
    "accessToken", "userId", "appId", "redirectUri",
  ]);

  try {
    const resp = await fetch(`${API_BASE}/fix`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": DEMO_API_KEY },
      body: JSON.stringify({
        error_code: errorCode,
        fix_category: analysisResult?.fix_category,
        access_token: accessToken || "",
        user_id: userId || null,
        app_id: appId || null,
        redirect_uri: redirectUri || null,
      }),
    });
    const result = await resp.json();
    chrome.action.setBadgeText({ text: "✓" });
    chrome.action.setBadgeBackgroundColor({ color: "#107c10" });
    return { success: true, result };
  } catch (err) {
    return { success: false, error: err.message };
  }
}
