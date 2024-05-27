# --- config.py

import os

# Discord Bot Token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")  # Get from environment variable

# Data Source URLs
AVIATION_WEATHER_URL = "https://aviationweather.gov/adds/dataserver_current/httpparam"
SPC_SOUNDING_URL = "https://www.spc.noaa.gov/cgi-bin/soundings/get_soundings.cgi"
GOES_SATELLITE_URL = "https://www.star.nesdis.noaa.gov/GOES/"
NWS_ALERTS_URL = "https://api.weather.gov/alerts"  # Replace with actual endpoint
RADAR_BASE_URL = "https://radar.weather.gov/ridge/lite/"

# Default Values
DEFAULT_REGION = "conus"
DEFAULT_SATELLITE_PRODUCT = 1  # GeoColor
DEFAULT_HOURS_BEFORE_NOW_METAR = 24
DEFAULT_HOURS_BEFORE_NOW_TAF = 24
DEFAULT_SOUNDING_DATA_SOURCE = "NAM"

# Bot Settings
COMMAND_PREFIX = "$"
STATUS_MESSAGE = "Weather Bot Online"

# Sounding Models
SOUNDING_MODELS = {
    "NAM": {"data_source": "NAM"},
    "GFS": {"data_source": "GFS"},
    "HRRR": {"data_source": "HRRR"},
    "RAP": {"data_source": "RAP"},  # If you've implemented RAP soundings
}

# RAP Products (Fill in actual codes and URLs from http://weather.cod.edu/analysis/)
RAP_PRODUCTS = {
    # Example: (replace with your actual RAP product codes and URLs)
    "analysis": {"url": "http://weather.cod.edu/analysis/analysis.gif"},
    "analysis_90": {"url": "http://weather.cod.edu/analysis/analysis_90.gif"},
}

# Geocoding Settings (Replace placeholders with actual keys if using a service)
GEOCODING_PROVIDER = "nominatim"
NOMINATIM_USER_AGENT = "your_weather_bot"
# GOOGLE_MAPS_API_KEY = ""
# OPENCAGE_API_KEY = ""

# Other API Keys (Add here if you integrate with other services)
# NWS_API_KEY = ""
# LIGHTNING_API_KEY = ""
# ...

# Map Overlay Settings
DEFAULT_MARKER_ICON = "default_marker.png"
LOGO_PATH = "logo.png"
