"""
Updating mapbox tiles
"""
import traceback
from pathlib import Path
import django
from django.conf import settings
from django.core.mail import EmailMessage
from services.landcover import Landcover
django.setup()

from jp2_to_tiff_conversion import jp2_to_tiff
from model_call import ModelCaller
from sentinel_download import SentinelDownload
from upload_to_mapbox import start_upload

LANDCOVER_URL = 'https://s3-eu-west-1.amazonaws.com\
/vito.landcover.global/2015/E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326.zip'


if __name__ == '__main__':
    try: 
        landcover = Landcover()
        if not landcover.forest_tiff.exists():
            raw_file = landcover.download_landcover(LANDCOVER_URL)
            # raw_file = landcover.data_path / 'landcover.zip'
            landcover.extract_file(raw_file, landcover.tif, landcover.data_path)
            landcover.copy_file(landcover.data_path / landcover.tif, landcover.forest_tiff)
            Path.unlink(landcover.data_path / landcover.tif)
        else:
            print(f'file {landcover.forest_tiff} already exists')
        # exit(0)
        sentinel_downloader = SentinelDownload()
        sentinel_downloader.process_download()
        sentinel_downloader.executor.shutdown()
        
        model_caller = ModelCaller()
        model_caller.start()
        
        jp2_to_tiff()
        uploader = start_upload()
        uploader.shutdown()

        model_caller.executor.shutdown()

    except Exception as error:
        pass
        EmailMessage(
            subject='Pep download issue',
            body=(
                f'Daemon can not download Sentinel2 data. Issue information listed bellow: '
                f'\n\n{str(error)}\n\n {"".join(traceback.format_tb(error.__traceback__))}'
            ),
            from_email=settings.EMAIL_HOST_USER,
            to=settings.EMAIL_ADMIN_MAILS
        ).send()
