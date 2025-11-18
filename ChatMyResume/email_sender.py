import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")


def send_email_to_user(subject: str, body: str) -> None:
    """
    Sends an email to yourself when the bot cannot answer.
    Uses Gmail + App Password (2FA) for reliability.
    """
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        # Fail silently in dev; log in prod
        print("Email not sent: missing EMAIL_ADDRESS or EMAIL_APP_PASSWORD.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    msg.set_content(body)

    # Gmail SMTP over SSL
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        smtp.send_message(msg)
