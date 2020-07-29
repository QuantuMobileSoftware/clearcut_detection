# Generated by Django 2.2.14 on 2020-07-20 06:57

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('clearcuts', '0006_01'),
    ]
    operations = [

        migrations.AlterField(
            model_name='tileinformation',
            name='model_tiff_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_b04_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_b08_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_b11_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_b12_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_b8a_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_clouds_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='source_tci_location',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='tileinformation',
            name='tile_date',
            field=models.DateField(default=datetime.datetime(2020, 7, 20, 6, 57, 9, 252519, tzinfo=utc)),
        ),
    ]
