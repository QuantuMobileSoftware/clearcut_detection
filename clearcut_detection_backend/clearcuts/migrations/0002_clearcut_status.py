# Generated by Django 2.2.14 on 2020-08-25 11:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('clearcuts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='clearcut',
            name='status',
            field=models.SmallIntegerField(default=0),
        ),
    ]
