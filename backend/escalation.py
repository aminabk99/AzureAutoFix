"""
AzureAutoFix — Escalation Message Generator
Generates pre-written admin emails and Teams messages.
"""


def generate_escalation_message(
    error_code: str,
    explanation: str,
    action_detail: str,
) -> dict:
    subject = f"[Action Required] Azure AD Error {error_code} — Fix Needed"

    body = f"""Hi,

I'm encountering an Azure Active Directory error that requires administrator action to resolve.

Error Code: {error_code}
What it means: {explanation}

Recommended fix: {action_detail}

Could you please apply this fix at your earliest convenience? This is blocking my access to [application/resource].

Thank you,
[Your name]
"""

    teams_message = (
        f"Hi @admin — I'm getting Azure AD error **{error_code}** "
        f"and need your help to fix it. "
        f"The issue is: _{explanation}_ "
        f"The fix is: _{action_detail}_ "
        f"Can you take a look when you get a chance? Thanks!"
    )

    return {
        "subject": subject,
        "body": body,
        "teams_message": teams_message,
    }
