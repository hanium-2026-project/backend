from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("parking", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="parkinglot",
            name="lot_width",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="parkinglot",
            name="lot_height",
            field=models.FloatField(default=0.0),
        ),
    ]
