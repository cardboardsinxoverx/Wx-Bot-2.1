# --- config.py

import os
# Discord Bot Token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")  # Get from environment variable

# Data Source URLs (Updated based on main_bot_script.txt)
AVIATION_WEATHER_URL = "https://aviationweather.gov/api/data/"  # Adjusted for METAR/TAF API
SPC_SOUNDING_URL = "https://weather.uwyo.edu/cgi-bin/sounding"  # Updated to Wyoming sounding archive
GOES_SATELLITE_URL = "https://whirlwind.aos.wisc.edu/~wxp/"  # Base URL for GOES images
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"  # Specific endpoint for active alerts
RADAR_BASE_URL = "https://tempest.aos.wisc.edu/radar/"  # Updated for the radar images

# Default Values
DEFAULT_REGION = "conus"
DEFAULT_SATELLITE_PRODUCT = 14  # CleanIR
DEFAULT_HOURS_BEFORE_NOW_METAR = 24
DEFAULT_HOURS_BEFORE_NOW_TAF = 24


# Bot Settings
COMMAND_PREFIX = "$"
STATUS_MESSAGE = "Weather Bot Online"

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
