from django.conf import settings
from django.core.mail import EmailMessage


def emaile_on_service_error(subject, body):
    EmailMessage(
        subject=subject,
        body=body,
        from_email=settings.EMAIL_HOST_USER,
        to=settings.EMAIL_ADMIN_MAILS
    ).send()
