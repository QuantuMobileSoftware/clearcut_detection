from configparser import ConfigParser

config = ConfigParser(allow_no_value=True)
config.read('gcp_config.ini')

area_tile_set = config.get('config', 'AREA_TILE_SET').split()
bands_to_download = config.get('config', 'BANDS_TO_DOWNLOAD').split()