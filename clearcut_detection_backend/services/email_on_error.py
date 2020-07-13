import traceback
from django.conf import settings
from django.core.mail import EmailMessage


def emaile_on_service_error(subject, error):
    EmailMessage(
        subject=subject,
        body=f'\n\n{str(error)}\n\n {"".join(traceback.format_tb(error.__traceback__))}',
        from_email=settings.EMAIL_HOST_USER,
        to=settings.EMAIL_ADMIN_MAILS
    ).send()
