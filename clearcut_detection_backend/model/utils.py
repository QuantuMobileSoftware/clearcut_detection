"""
Model's helpers
"""
import io
import os.path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive.file']


def weights_exists_or_download(path, file_id):
    if not os.path.exists(path):
        creds_file = os.environ.get('CREDENTIAL_FILE')
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=SCOPES)

        service = build('drive', 'v3', credentials=creds)
        request = service.files().get_media(fileId=file_id)

        fh = io.FileIO('unet_v4.pth', mode='wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f'Download {int(status.progress() * 100)}')

    return path
