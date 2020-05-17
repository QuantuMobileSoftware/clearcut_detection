import django
from django.conf import settings
django.setup()
from clearcuts.geojson_save import save
import easyargs

@easyargs
def save_poly(path):
    save(path)

if __name__ == "__main__":
    
    save_poly()
