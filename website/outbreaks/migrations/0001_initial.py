# Generated manually for the portable project artifact.
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='AirTrafficAggregate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(db_index=True)),
                ('year', models.IntegerField(db_index=True)),
                ('month', models.IntegerField(db_index=True)),
                ('state_name', models.CharField(db_index=True, max_length=80)),
                ('dest_city_name', models.CharField(blank=True, max_length=160)),
                ('dest_airport', models.CharField(blank=True, db_index=True, max_length=16)),
                ('dest_lat', models.FloatField(blank=True, null=True)),
                ('dest_lon', models.FloatField(blank=True, null=True)),
                ('origin_country', models.CharField(db_index=True, max_length=120)),
                ('origin_country_code', models.CharField(blank=True, db_index=True, max_length=8)),
                ('flights', models.FloatField(default=0)),
                ('passengers', models.FloatField(default=0)),
                ('freight', models.FloatField(default=0)),
                ('mail', models.FloatField(default=0)),
                ('payload', models.FloatField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='DetectionAggregate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(db_index=True)),
                ('year', models.IntegerField(db_index=True)),
                ('month', models.IntegerField(db_index=True)),
                ('common_name', models.CharField(blank=True, db_index=True, max_length=160)),
                ('state_name', models.CharField(db_index=True, max_length=80)),
                ('county_name', models.CharField(blank=True, db_index=True, max_length=160)),
                ('count', models.FloatField(default=0)),
                ('lat', models.FloatField(blank=True, null=True)),
                ('lon', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PortTraffic',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField(db_index=True)),
                ('port_name', models.CharField(db_index=True, max_length=180)),
                ('state', models.CharField(db_index=True, max_length=40)),
                ('foreign_inbound_loaded', models.FloatField(default=0)),
                ('foreign_outbound_loaded', models.FloatField(default=0)),
                ('total_foreign_loaded', models.FloatField(default=0)),
                ('grand_total_loaded', models.FloatField(default=0)),
            ],
        ),
        migrations.AddIndex(model_name='airtrafficaggregate', index=models.Index(fields=['origin_country', 'year', 'month'], name='outbreaks_a_orig_426b0d_idx')),
        migrations.AddIndex(model_name='airtrafficaggregate', index=models.Index(fields=['state_name', 'year', 'month'], name='outbreaks_a_state_1c03a6_idx')),
        migrations.AddIndex(model_name='detectionaggregate', index=models.Index(fields=['state_name', 'year', 'month'], name='outbreaks_d_state_cc71da_idx')),
        migrations.AddIndex(model_name='porttraffic', index=models.Index(fields=['state', 'year'], name='outbreaks_p_state_91dc97_idx')),
    ]
