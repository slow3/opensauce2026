"""
Email Sender — sends guest delivery emails via SMTP.
Uses the existing smtp.json config format from AutoRC as reference.
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger("email")


def _smtp_password(config: dict) -> str:
    """Read SMTP password from environment variable — never hardcode credentials."""
    return os.environ.get("SCANOPS_SMTP_PASSWORD", "")


def send_delivery_email(config: dict, guest: dict, rig: int, session_id: str, outputs: dict):
    email_cfg  = config["email"]
    guest_name = guest.get("name", "Guest")
    guest_email = guest.get("email", "")

    if not guest_email:
        log.warning(f"No email for {guest_name} — skipping")
        return

    viewer_html  = outputs.get("viewer_html")
    mesh_fbx     = outputs.get("mesh_fbx")
    prusa_file   = outputs.get("prusa_file")

    subject = f"Your Open Sauce 2026 scan is ready, {guest_name.split()[0]}!"

    if rig == 1:
        body = _rig1_email_body(guest_name, session_id, viewer_html)
    else:
        body = _rig2_email_body(guest_name, session_id, viewer_html, mesh_fbx, prusa_file)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{email_cfg['sender_name']} <{email_cfg['sender_email']}>"
    msg["To"]      = guest_email
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(email_cfg["smtp_host"], email_cfg["smtp_port"]) as server:
            if email_cfg.get("use_tls"):
                server.starttls()
            password = _smtp_password(config)
            if password:
                server.login(email_cfg["sender_email"], password)
            server.sendmail(email_cfg["sender_email"], guest_email, msg.as_string())
        log.info(f"Email sent to {guest_email}")
    except Exception as e:
        log.error(f"Failed to send email to {guest_email}: {e}")
        raise


def _rig1_email_body(name: str, session_id: str, viewer_html: str | None) -> str:
    viewer_link = viewer_html or "#"
    return f"""
<!DOCTYPE html>
<html>
<body style="background:#0a0a0f;color:#f0f0f8;font-family:Inter,Arial,sans-serif;padding:40px;max-width:600px;margin:0 auto;">
  <h1 style="color:#1D9E75;font-size:24px;margin-bottom:8px;">Your Gaussian Splat is ready!</h1>
  <p style="color:#f0f0f880;margin-bottom:32px;">Session #{session_id}</p>

  <p>Hey {name.split()[0]},</p>
  <p>Your 3D Gaussian Splat from Open Sauce 2026 has been processed and is ready to view.</p>

  <div style="text-align:center;margin:32px 0;">
    <a href="{viewer_link}"
       style="background:#1D9E75;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:600;font-size:16px;">
      View Your 3D Splat
    </a>
  </div>

  <p style="color:#f0f0f860;font-size:13px;">
    Captured at Open Sauce 2026 by Scan The Wild.<br>
    Questions? Reply to this email.
  </p>
</body>
</html>
"""


def _rig2_email_body(name: str, session_id: str, viewer_html: str | None,
                     mesh_fbx: str | None, prusa_file: str | None) -> str:
    viewer_link  = viewer_html or "#"
    print_status = "Your 3D print is in the queue — pick it up at the Scan The Wild booth!" if prusa_file else ""

    return f"""
<!DOCTYPE html>
<html>
<body style="background:#0a0a0f;color:#f0f0f8;font-family:Inter,Arial,sans-serif;padding:40px;max-width:600px;margin:0 auto;">
  <h1 style="color:#1D9E75;font-size:24px;margin-bottom:8px;">Your 3D scan is ready!</h1>
  <p style="color:#f0f0f880;margin-bottom:32px;">Session #{session_id}</p>

  <p>Hey {name.split()[0]},</p>
  <p>Your object scan from Open Sauce 2026 is complete. You have a Gaussian Splat viewer and a full mesh file.</p>

  <div style="text-align:center;margin:32px 0;">
    <a href="{viewer_link}"
       style="background:#1D9E75;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:600;font-size:16px;">
      View Your 3D Splat
    </a>
  </div>

  {f'<p style="background:#1D9E7518;border:1px solid #1D9E7544;border-radius:8px;padding:16px;color:#1D9E75;">{print_status}</p>' if print_status else ''}

  <p style="color:#f0f0f860;font-size:13px;">
    Captured at Open Sauce 2026 by Scan The Wild.<br>
    Questions? Reply to this email.
  </p>
</body>
</html>
"""
