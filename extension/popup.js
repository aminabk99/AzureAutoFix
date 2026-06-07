/**
 * AzureAutoFix — Popup Script
 * Reads stored error data from chrome.storage and renders the appropriate UI state.
 */

const API_BASE = "http://localhost:8000";

const USER_STEPS = {
  prompt_reauth: [
    "Click Sign out from the application.",
    "Clear your browser cookies for login.microsoftonline.com.",
    "Sign in again with your correct username and password.",
    "If you forgot your password, click 'Forgot my password'.",
  ],
  trigger_interactive_login: [
    "Close all browser tabs for this application.",
    "Open a new private/incognito window.",
    "Navigate back to the app and sign in again.",
    "Complete any MFA prompt if one appears.",
  ],
  trigger_consent_flow: [
    "When the permissions dialog appears, click Accept.",
    "If no prompt appears, try adding ?prompt=consent to the app URL.",
    "If blocked, your admin may need to grant org-wide consent.",
  ],
  redirect_correct_tenant: [
    "Sign out of your current account.",
    "Sign back in using your work email (e.g. you@yourcompany.com).",
    "If the issue persists, contact your IT admin to register the app.",
  ],
  retry: [
    "Wait 10 seconds.",
    "Try your action again — this was a transient Azure AD error.",
    "If it persists more than 3 times, contact Microsoft Support.",
  ],
};

function getBadgeClass(fixCategory) {
  const map = {
    admin_auto: "badge-auto",
    user: "badge-user",
    admin_escalate: "badge-escalate",
    retry: "badge-retry",
  };
  return map[fixCategory] || "badge-user";
}

function getBadgeLabel(fixCategory) {
  const map = {
    admin_auto: "Auto-Fix Available",
    user: "User Action",
    admin_escalate: "Admin Required",
    retry: "Retry",
  };
  return map[fixCategory] || fixCategory;
}

async function loadState() {
  const data = await chrome.storage.local.get(["lastError", "lastAnalysis"]);

  if (!data.lastError || !data.lastAnalysis) {
    document.getElementById("no-error").style.display = "block";
    return;
  }

  const { lastError: errorCode, lastAnalysis: analysis } = data;
  document.getElementById("error-state").style.display = "block";

  // Fill error info
  document.getElementById("error-code-display").textContent = errorCode;
  document.getElementById("error-message").textContent = analysis.user_message;

  const badge = document.getElementById("fix-badge");
  badge.textContent = getBadgeLabel(analysis.fix_category);
  badge.className = `badge ${getBadgeClass(analysis.fix_category)}`;

  // Reasoning toggle
  const reasoningBox = document.getElementById("reasoning-box");
  reasoningBox.textContent =
    `Category: ${analysis.fix_category}\nConfidence: ${(analysis.confidence * 100).toFixed(0)}%\n\n${analysis.reasoning}`;

  document.getElementById("toggle-reasoning").onclick = () => {
    const visible = reasoningBox.style.display !== "none";
    reasoningBox.style.display = visible ? "none" : "block";
    document.getElementById("toggle-reasoning").textContent = visible
      ? "Show reasoning ▼"
      : "Hide reasoning ▲";
  };

  // Render state-specific card
  const fc = analysis.fix_category;

  if (fc === "admin_auto") {
    document.getElementById("admin-fix-card").style.display = "block";
    document.getElementById("auto-fix-btn").onclick = () => doFix(errorCode, analysis);
  } else if (fc === "user" || fc === "retry") {
    const card = document.getElementById("user-fix-card");
    card.style.display = "block";
    const steps = USER_STEPS[analysis.action] || USER_STEPS["retry"];
    const ol = document.getElementById("user-steps");
    steps.forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      ol.appendChild(li);
    });
  } else if (fc === "admin_escalate") {
    document.getElementById("escalate-card").style.display = "block";
    loadEscalation(errorCode);
  }
}

async function loadEscalation(errorCode) {
  try {
    const resp = await fetch(`${API_BASE}/escalate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error_input: errorCode }),
    });
    const data = await resp.json();
    document.getElementById("escalate-text").value = data.body;
    document.getElementById("copy-escalate-btn").onclick = () => {
      navigator.clipboard.writeText(data.body);
      document.getElementById("copy-escalate-btn").textContent = "Copied!";
      setTimeout(() => {
        document.getElementById("copy-escalate-btn").textContent = "Copy message";
      }, 2000);
    };
  } catch (e) {
    document.getElementById("escalate-text").value = "Could not load escalation message. Is the backend running?";
  }
}

async function doFix(errorCode, analysis) {
  const accessToken = document.getElementById("access-token-input").value.trim();
  const userId = document.getElementById("user-id-input").value.trim();
  const appId = document.getElementById("app-id-input").value.trim();
  const redirectUri = document.getElementById("redirect-uri-input").value.trim();

  if (!accessToken) {
    alert("Please enter your MS Graph access token.");
    return;
  }

  await chrome.storage.local.set({ accessToken, userId, appId, redirectUri });

  const progressCard = document.getElementById("fix-progress");
  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");
  progressCard.style.display = "block";

  const steps = [
    [20, "Connecting to MS Graph API..."],
    [50, `Applying fix for ${errorCode}...`],
    [80, "Verifying change..."],
    [100, "Done!"],
  ];

  for (const [pct, label] of steps) {
    progressFill.style.width = `${pct}%`;
    progressLabel.textContent = label;
    await new Promise((r) => setTimeout(r, 700));
  }

  try {
    const resp = await fetch(`${API_BASE}/fix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        error_code: errorCode,
        fix_category: analysis.fix_category,
        access_token: accessToken,
        user_id: userId || null,
        app_id: appId || null,
        redirect_uri: redirectUri || null,
      }),
    });
    const result = await resp.json();
    if (result.success) {
      progressLabel.textContent = `✅ ${result.details}`;
      progressFill.style.background = "#107c10";
    } else {
      progressLabel.textContent = `❌ Fix failed: ${result.detail || "Unknown error"}`;
    }
  } catch (e) {
    progressLabel.textContent = `❌ Error: ${e.message}`;
  }
}

document.addEventListener("DOMContentLoaded", loadState);
