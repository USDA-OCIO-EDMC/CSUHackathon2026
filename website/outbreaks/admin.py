from django.contrib import admin
from .models import AirTrafficAggregate, DetectionAggregate, PortTraffic

@admin.register(DetectionAggregate)
class DetectionAggregateAdmin(admin.ModelAdmin):
    list_display = ("date", "common_name", "state_name", "county_name", "count")
    list_filter = ("year", "month", "state_name", "common_name")
    search_fields = ("county_name", "state_name", "common_name")

@admin.register(AirTrafficAggregate)
class AirTrafficAggregateAdmin(admin.ModelAdmin):
    list_display = ("date", "origin_country", "state_name", "dest_airport", "passengers", "freight")
    list_filter = ("year", "month", "state_name", "origin_country")
    search_fields = ("origin_country", "dest_city_name", "dest_airport")

@admin.register(PortTraffic)
class PortTrafficAdmin(admin.ModelAdmin):
    list_display = ("year", "port_name", "state", "foreign_inbound_loaded", "total_foreign_loaded")
    list_filter = ("year", "state")
    search_fields = ("port_name", "state")
