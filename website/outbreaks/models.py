from django.db import models

class DetectionAggregate(models.Model):
    """Fruit fly detection rows, reduced to points so maps stay fast."""
    date = models.DateField(db_index=True)
    year = models.IntegerField(db_index=True)
    month = models.IntegerField(db_index=True)
    common_name = models.CharField(max_length=160, blank=True, db_index=True)
    state_name = models.CharField(max_length=80, db_index=True)
    county_name = models.CharField(max_length=160, blank=True, db_index=True)
    count = models.FloatField(default=0)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["state_name", "year", "month"])]

class AirTrafficAggregate(models.Model):
    """Inbound international traffic to US destination airports, aggregated monthly."""
    date = models.DateField(db_index=True)
    year = models.IntegerField(db_index=True)
    month = models.IntegerField(db_index=True)
    state_name = models.CharField(max_length=80, db_index=True)
    dest_city_name = models.CharField(max_length=160, blank=True)
    dest_airport = models.CharField(max_length=16, blank=True, db_index=True)
    dest_lat = models.FloatField(null=True, blank=True)
    dest_lon = models.FloatField(null=True, blank=True)
    origin_country = models.CharField(max_length=120, db_index=True)
    origin_country_code = models.CharField(max_length=8, blank=True, db_index=True)
    flights = models.FloatField(default=0)
    passengers = models.FloatField(default=0)
    freight = models.FloatField(default=0)
    mail = models.FloatField(default=0)
    payload = models.FloatField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=["origin_country", "year", "month"]),
            models.Index(fields=["state_name", "year", "month"]),
        ]

class PortTraffic(models.Model):
    year = models.IntegerField(db_index=True)
    port_name = models.CharField(max_length=180, db_index=True)
    state = models.CharField(max_length=40, db_index=True)
    foreign_inbound_loaded = models.FloatField(default=0)
    foreign_outbound_loaded = models.FloatField(default=0)
    total_foreign_loaded = models.FloatField(default=0)
    grand_total_loaded = models.FloatField(default=0)

    class Meta:
        indexes = [models.Index(fields=["state", "year"])]
