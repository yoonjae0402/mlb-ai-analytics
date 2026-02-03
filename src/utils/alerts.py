import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manages alerts for cost thresholds and errors via email.
    """
    
    def __init__(self):
        self.email_enabled = all([
            os.getenv('ALERT_EMAIL_FROM'),
            os.getenv('ALERT_EMAIL_TO'),
            os.getenv('ALERT_EMAIL_PASSWORD')
        ])
    
    def check_cost_threshold(self, daily_cost: float, threshold: float = 1.00) -> bool:
        """
        Check if daily cost exceeds threshold and send alert.
        
        Args:
            daily_cost: Total cost for the day
            threshold: Alert threshold in dollars
            
        Returns:
            True if alert was sent
        """
        if daily_cost > threshold:
            message = f"""
⚠️ COST ALERT ⚠️

Daily API cost has exceeded the threshold!

Current Cost: ${daily_cost:.2f}
Threshold: ${threshold:.2f}
Overage: ${daily_cost - threshold:.2f}

Please review your API usage.
            """.strip()
            
            logger.warning(f"Cost threshold exceeded: ${daily_cost:.2f} > ${threshold:.2f}")
            
            # Send email alert
            self.send_email_alert("MLB Pipeline: Cost Alert", message)
            
            return True
        
        return False
    
    def send_email_alert(self, subject: str, message: str) -> bool:
        """
        Send email alert.
        """
        if not self.email_enabled:
            logger.debug("Email alerts not configured")
            return False
        
        try:
            from_email = os.getenv('ALERT_EMAIL_FROM')
            to_email = os.getenv('ALERT_EMAIL_TO')
            password = os.getenv('ALERT_EMAIL_PASSWORD')
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_error_alert(self, error_message: str, context: dict):
        """
        Send alert for pipeline errors.
        """
        message = f"""
❌ PIPELINE ERROR ❌

An error occurred during pipeline execution:

Error: {error_message}

Context:
{chr(10).join(f"  {k}: {v}" for k, v in context.items())}

Please investigate immediately.
        """.strip()
        
        self.send_email_alert("MLB Pipeline: Error Alert", message)
