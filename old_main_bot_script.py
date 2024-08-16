## botsucks

# Imports and Setup
import sys
print(sys.path)
import discord
from discord.ext import commands
import requests
import datetime
import pytz
from bs4 import BeautifulSoup  # Instead of 'import BeautifulSoup'
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sharppy
import sharppy.plot.skew as skew
import cartopy.crs as ccrs
import os
import json
import logging
from geopy.geocoders import Nominatim
import astropy.coordinates as coord
from astropy.time import Time
from timezonefinder import TimezoneFinder
import ephem
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import geocoder
import subprocess  # Add this line at the beginning
# ... (your existing imports and code)

import json

def load_cache(cache_name):
    try:
        with open(f"{cache_name}_cache.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the cache file doesn't exist


try:
    subprocess.run(["python", "main_bot_script.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running script: {e.output}")

# Load Configuration
import config

# Configure logging
logging.basicConfig(
    filename="weather_bot.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}') # idk i hope this works

# Data Storage (Caching)
metar_cache = {}
taf_cache = {}
alert_cache = {}

# Load Cache Data on Startup
metar_cache = load_cache("metar")  # keep getting and error with this line sometimes ohh
taf_cache = load_cache("taf") # guess maybe this one too
alert_cache = load_cache("alert") # this is gay

# Helper Functions for Caching (load_cache, save_cache)
def load_cache(cache_type):
    try:
        with open(f"{cache_type}_cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache_type, data):
    with open(f"{cache_type}_cache.json", "w") as f:
        json.dump(data, f, indent=2)

# --- on_message Event Handler ---
@bot.event
async def on_message(message):
    if message.author == bot.user:  # Don't respond to self
        return


    await bot.process_commands(message)  # Process bot commands

# Helper Functions for Caching (load_cache, save_cache)
def load_cache(cache_type):
    try:
        with open(f"{cache_type}_cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache_type, data):
    with open(f"{cache_type}_cache.json", "w") as f:
        json.dump(data, f, indent=2)

# Data Storage (Caching)
metar_cache = {}
taf_cache = {}
alert_cache = {}

# Load Cache Data on Startup
metar_cache = load_cache("metar")
taf_cache = load_cache("taf")
alert_cache = load_cache("alert")

# --- METAR Command ---
@bot.command(aliases=["wx"])
async def metar(ctx, airport_code: str, hours_ago: int = config.DEFAULT_HOURS_BEFORE_NOW_METAR):
    """Fetches METAR for the specified airport code. Optionally specify hours ago for historical data."""
    try:
        if hours_ago < 0:
            raise ValueError("Invalid hours ago. Please enter a non-negative number.")

        # --- 1. Construct URL ---
        url = f"{config.AVIATION_WEATHER_URL}?dataSource=metars&requestType=retrieve&format=xml&stationString={airport_code.upper()}&hoursBeforeNow={hours_ago}"  # Uppercase for consistency

        # --- 2. Fetch Data ---
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request fails

        # --- 3. Parse METAR ---
        soup = BeautifulSoup(response.content, 'xml')
        metar_data = soup.find('raw_text').text

        if not metar_data:
            raise ValueError("METAR data not found.")

        # --- 4. Extract Time (for Historical METARs) ---
        if hours_ago > 0:
            metar_time = soup.find('observation_time').text
        else:
            metar_time = None  # Current METAR doesn't need time in output

        # --- 5. Prepare and Send Response ---
        if metar_time:
            message = f"METAR for {airport_code} ({metar_time}Z): {metar_data}"
        else:
            message = f"METAR for {airport_code}: {metar_data}"

        # Optionally, use discord.Embed for better formatting
        embed = discord.Embed(title=f"METAR for {airport_code}", description=metar_data)
        if metar_time:
            embed.set_footer(text=f"Observation Time: {metar_time}Z")
        await ctx.send(embed=embed)

        # --- 6. Update Cache ---
        if hours_ago == 0: # Only cache current METARs
            if airport_code not in metar_cache:
                metar_cache[airport_code] = []
            metar_cache[airport_code].append({
                "time": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data": metar_data
            })
            save_cache("metar", metar_cache)

        logging.info(f"User {ctx.author} requested METAR for {airport_code} (hours ago: {hours_ago})")

    # --- 7. Error Handling ---
    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing METAR for {airport_code}: {e}")
        logging.error(f"Error retrieving/parsing METAR for {airport_code}: {e}")

# --- TAF Command ---
@bot.command()
async def skewt(ctx, sounding_type: str, *args):
    """Fetches TAF for the specified airport code."""
    try:
        # --- 1. Input Handling and Caching ---
        airport_code = airport_code.upper()

        # Check for cached TAF
        if airport_code in taf_cache:
            for entry in taf_cache[airport_code]:
                if entry['time'] == (datetime.datetime.utcnow() - datetime.timedelta(hours=config.DEFAULT_HOURS_BEFORE_NOW_TAF)).strftime("%Y-%m-%dT%H:%M:%SZ"):
                    logging.info(f"User {ctx.author} requested TAF for {airport_code} (cached)")
                    await ctx.send(f"TAF for {airport_code} (cached): {entry['data']}")
                    return

        # --- 2. Construct URL ---
        url = f"{config.AVIATION_WEATHER_URL}?dataSource=tafs&requestType=retrieve&format=xml&stationString={airport_code}&hoursBeforeNow={config.DEFAULT_HOURS_BEFORE_NOW_TAF}"

        # --- 3. Fetch Data ---
        response = requests.get(url)
        response.raise_for_status()

        # --- 4. Parse TAF ---
        soup = BeautifulSoup(response.content, 'xml')
        taf_data = soup.find('raw_text').text

        if not taf_data:
            raise ValueError("TAF data not found.")

        # --- 5. Extract Time (for Historical TAFs) ---
        if config.DEFAULT_HOURS_BEFORE_NOW_TAF > 0:
            taf_time = soup.find('issue_time').text
        else:
            taf_time = None  # Current TAF doesn't need time in output

        # --- 6. Prepare and Send Response ---
        if taf_time:
            message = f"TAF for {airport_code} ({taf_time}Z): {taf_data}"
        else:
            message = f"TAF for {airport_code}: {taf_data}"

        # Optionally, use discord.Embed for better formatting
        embed = discord.Embed(title=f"TAF for {airport_code}", description=taf_data)
        if taf_time:
            embed.set_footer(text=f"Issue Time: {taf_time}Z")
        await ctx.send(embed=embed)

        # --- 7. Update Cache (only for current TAFs) ---
        if config.DEFAULT_HOURS_BEFORE_NOW_TAF == 0:
            if airport_code not in taf_cache:
                taf_cache[airport_code] = []
            taf_cache[airport_code].append({
                "time": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data": taf_data
            })
            save_cache("taf", taf_cache)

        logging.info(f"User {ctx.author} requested TAF for {airport_code}")

    # --- 8. Error Handling ---
    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing TAF for {airport_code}: {e}")
        logging.error(f"Error retrieving/parsing TAF for {airport_code}: {e}")

# --- Skew-T Command ---
@bot.command()
async def skewt(ctx, sounding_type: str, *args):
    """Fetches sounding data and generates a Skew-T diagram using SHARPpy.

    Usage:
    - $skewt observed <station_code>
    - $skewt fcst <model> <forecast_hour> <station_code>
    """
    try:
        # 1. Input Validation and Parameter Setup
        station_code = args[-1].upper()
        valid_sounding_types = ["observed", "fcst"]

        if sounding_type.lower() not in valid_sounding_types:
            raise ValueError("Invalid sounding type. Please specify 'observed' or 'fcst'.")

        if sounding_type.lower() == "observed":
            model = config.DEFAULT_SOUNDING_DATA_SOURCE
            forecast_hour = 0
        else:  # Forecast
            if len(args) < 3:
                raise ValueError("Missing model and/or forecast hour for forecast Skew-T.")
            model = args[0].upper()
            forecast_hour = int(args[1])

            if model not in config.SOUNDING_MODELS:
                raise ValueError(f"Invalid model. Valid options are: {', '.join(config.SOUNDING_MODELS.keys())}")
            if forecast_hour < 0:
                raise ValueError("Forecast hour must be a non-negative integer.")

        # Construct Sounding URL
        model_params = config.SOUNDING_MODELS[model]
        if model == "HRRR":
            sounding_url = f"{config.SPC_SOUNDING_URL}?data_source=HRRR&fcst_hour={forecast_hour}&airport={station_code}"
        else:
            model_params['fcst_len'] = forecast_hour  # Set forecast length for other models
            sounding_url = f"{config.SPC_SOUNDING_URL}?{'&'.join([f'{k}={v}' for k, v in model_params.items()])}&airport={station_code}"


        # 2. Fetch Sounding Data
        response = requests.get(sounding_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        sounding_data = soup.find("pre").text.strip()

        if not sounding_data:
            raise ValueError("Sounding data not found.")

        # 3. Parse and Generate Skew-T
        profile = sharppy.Profile.from_sounding(sounding_data)
        fig = plt.figure(figsize=(8, 8))
        skew.plot(profile, showparcel=True)

        # Add logo to Skew-T diagram
        # add_map_overlay(fig.gca(), logo_path="logo.png")

        # 4. Prepare and Send Response
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        sounding_type_str = "forecast" if forecast_hour > 0 else "observed"
        await ctx.send(file=discord.File(buffer, f"skewt_{station_code}_{model}_{sounding_type_str}.png"))

        logging.info(f"User {ctx.author} requested Skew-T for {station_code} ({model} - {sounding_type_str})")

    # 5. Error Handling
    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing Skew-T data for {station_code} using {model}: {e}")
        logging.error(f"Error retrieving/parsing Skew-T data for {station_code}: {e}")

# --- Satellite Command ---
@bot.command()
async def sat(ctx, region: str = config.DEFAULT_REGION, product_code: int = config.DEFAULT_SATELLITE_PRODUCT):
    """Fetches satellite image for the specified region and product code.

    Regions:
      - conus (default)
      - fulldisk
      - mesosector1
      - mesosector2
      - tropicalatlantic
      - tropicalpacific

    Product Codes (CONUS, Tropical Atlantic, Tropical Pacific):
      - 1: GeoColor (True Color)
      - 2: Red Visible
      - 14: Clean Longwave Infrared Window
      - 9: Mid-level Water Vapor

    Product Codes (Full Disk, Mesosectors):
      - 1: GeoColor (True Color)
      - 2: Red Visible
      - 13: Clean Longwave Infrared Window
    """

    try:
        region = region.lower()
        valid_regions = ["conus", "fulldisk", "mesosector1", "mesosector2", "tropicalatlantic", "tropicalpacific"]

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")

        # Product codes for different regions
        product_codes = {
            "conus": {
                1: "GeoColor (True Color)",
                2: "Band 2 - Red Visible",
                14: "Band 14 - Clean Longwave Infrared Window",
                9: "Band 9 - Mid-level Water Vapor",
            },
            "fulldisk": {
                1: "GeoColor (True Color)",
                2: "Band 2 - Red Visible",
                13: "Band 13 - Clean Longwave Infrared Window",
            },
            "mesosector1": {
                1: "GeoColor (True Color)",
                2: "Band 2 - Red Visible",
                13: "Band 13 - Clean Longwave Infrared Window",
            },
            "mesosector2": {
                1: "GeoColor (True Color)",
                2: "Band 2 - Red Visible",
                13: "Band 13 - Clean Longwave Infrared Window",
            },
            "tropicalatlantic": {
                1: "GeoColor (True Color)",
                2: "Band 2 - Red Visible",
                14: "Band 14 - Clean Longwave Infrared Window",
                9: "Band 9 - Mid-level Water Vapor",
            },
            "tropicalpacific": {
                1: "GeoColor (True Color)",
                2: "Band 2 - Red Visible",
                14: "Band 14 - Clean Longwave Infrared Window",
                9: "Band 9 - Mid-level Water Vapor",
            },
        }

        # Error handling for invalid product code
        if product_code not in product_codes[region]:
            raise ValueError(f"Invalid product code for {region}. Valid codes are: {', '.join(map(str, product_codes[region].keys()))}")

        # Construct the image URL based on the region and product code
        band_or_product = "GEOCOLOR" if product_code == 1 else f"C{product_code}"

        if region == "conus":
            image_url = f"{config.GOES_SATELLITE_URL}conus_band.php?sat=G16&band={band_or_product}&length=24"
        elif region == "fulldisk":
            image_url = f"{config.GOES_SATELLITE_URL}fulldisk.php?sat=G16&band={band_or_product}&length=24"
        elif region in ["tropicalatlantic", "tropicalpacific"]:
            image_url = f"{config.GOES_SATELLITE_URL}{region}_band.php?sat=G16&band={band_or_product}&length=24"
        else:  # For mesosectors
            image_url = f"{config.GOES_SATELLITE_URL}mesosector.php?sat=G16&sector={region}&band={band_or_product}&length=24"

        # Fetch satellite image
        response = requests.get(image_url)
        response.raise_for_status()

        # Load and process the image
        img = Image.open(BytesIO(response.content))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.axis('off')

        # Add the logo
        add_map_overlay(ax, logo_path="logo.png")  # Make sure you have a logo.png file

        # Save plot to BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        await ctx.send(file=discord.File(buffer, f"{product_codes[region][product_code]}.png"))
        logging.info(f"User {ctx.author} requested satellite image for {region} (product code {product_code})")

    except (requests.exceptions.RequestException, AttributeError, ValueError, KeyError) as e:
        await ctx.send(f"Error retrieving/parsing satellite imagery: {e}")
        logging.error(f"Error retrieving/parsing satellite imagery: {e}")

# --- Astronomy Command ---
@bot.command()
async def astro(ctx, location: str = None):
    """Provides sunrise, sunset, and moon phase information for a given location. this section will probably need some work fam. you just type in a city, i guess the country shouldn't matter."""
    if not location:
        await ctx.send("Please provide a location (e.g., '$astro New York City')")
        return

    try:
        # Geocode location to get coordinates
        geolocator = Nominatim(user_agent="weather-bot")
        loc = geolocator.geocode(location)
        if not loc:
            raise ValueError("Location not found.")

        # Determine time zone
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lng=loc.longitude, lat=loc.latitude)

        # Get current time in the specified time zone
        now = datetime.datetime.now(pytz.timezone(timezone))

        # Calculate sunrise and sunset using PyEphem
        obs = ephem.Observer()
        obs.lat = str(loc.latitude)
        obs.long = str(loc.longitude)
        obs.date = now
        sun = ephem.Sun()
        sunrise = obs.next_rising(sun).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))
        sunset = obs.next_setting(sun).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))

        # Calculate moon phase
        moon = ephem.Moon()
        moon.compute(now)
        moon_phase = moon.phase

        # Format the results
        response = f"Astronomy information for {location}:\n\n"
        response += f"**Sunrise:** {sunrise.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Sunset:** {sunset.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Moon Phase:** {moon_phase:.1f}% (Illuminated)"

        await ctx.send(response)
        logging.info(f"User {ctx.author} requested astronomy information for {location}")
    except (GeocoderTimedOut, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving astronomy information: {e}")
        logging.error(f"Error retrieving astronomy information for {location}: {e}")

# --- Radar Command ---
@bot.command()
async def radar(ctx, location: str = None):
    """Fetches radar imagery from NWS for a specified location or the user's location (if set). so this is sorta like pulling a metar. you'd just type $radar ffc or the last three letters of the radar code. remember they aren't always the same as the airport, I'm just lucky I don't need to remember two codes."""
    try:
        if location:
            # Geocode provided location
            geolocator = Nominatim(user_agent="weather-bot")
            loc = geolocator.geocode(location)
            if not loc:
                raise ValueError("Location not found.")
            latitude, longitude = loc.latitude, loc.longitude
        else:
            # Attempt to use user's location from Discord profile
            if ctx.author.nick:  # Check for nickname
                location_str = ctx.author.nick
            else:
                location_str = ctx.author.name
            loc = geolocator.geocode(location_str)
            if not loc:
                raise ValueError("Location not found. Please provide a location or set your nickname/username in Discord to your location.")
            latitude, longitude = loc.latitude, loc.longitude

        # Fetch radar image from NWS
        radar_url = f"https://radar.weather.gov/ridge/lite/{latitude}_{longitude}_0.png"
        response = requests.get(radar_url)
        response.raise_for_status()

        # Load image
        img = Image.open(BytesIO(response.content))
        img_np = np.array(img)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot image
        ax.imshow(img_np, origin='upper', extent=(-125, -67, 24, 50), transform=ccrs.PlateCarree())

        # Plot marker with custom icon
        add_map_overlay(ax, latitude, longitude)

        # Additional map features (e.g., coastlines, borders)
        ax.coastlines()
        ax.add_feature(cartopy.feature.STATES)

        # Add latitude and longitude labels
        ax.set_xticks(np.arange(-120, -65, 5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(25, 51, 5), crs=ccrs.PlateCarree())
        lon_formatter = cartopy.mpl.ticker.LongitudeFormatter()
        lat_formatter = cartopy.mpl.ticker.LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        # Save plot to BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        await ctx.send(file=discord.File(buffer, f"radar_{location}.png"))
        logging.info(f"User {ctx.author} requested radar for {location}")

    except (requests.exceptions.RequestException, AttributeError, ValueError, geocoder.GeocoderTimedOut) as e:
        await ctx.send(f"Error retrieving/parsing radar imagery for {location}: {e}")
        logging.error(f"Error retrieving/parsing radar imagery for {location}: {e}")

# --- overlay that wont work ---
def add_map_overlay(ax, lat=None, lon=None, icon_path=None, logo_path="logo.png", zoom=0.1):
    """Adds a marker (if lat/lon provided) and a logo to the map image.

    Args:
        ax: The Matplotlib Axes object representing the map.
        lat (float, optional): Latitude of the marker (if needed). Defaults to None.
        lon (float, optional): Longitude of the marker (if needed). Defaults to None.
        icon_path (str, optional): Path to the marker icon image. Defaults to None.
        logo_path (str, optional): Path to the logo image file. Defaults to "logo.png".
        zoom (float, optional): Zoom level for the marker icon. Defaults to 0.1.
    """

    # 1. Add Location Marker (if coordinates are provided)
    if lat is not None and lon is not None:
        try:
            if icon_path:  # Use custom icon if provided
                img = Image.open(icon_path)
            else:  # Use default marker icon
                img = Image.open("default_marker.png")  # Replace with your default icon
        except (FileNotFoundError, PIL.UnidentifiedImageError):
            logging.warning("Error loading marker icon. Using default.")
            img = Image.open("default_marker.png")

        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                           boxcoords="offset points", pad=0, zorder=10)  # Higher zorder to place above other elements
        ax.add_artist(ab)
        ax.annotate(f"({lat:.2f}, {lon:.2f})", (lon, lat), xytext=(3, 3), textcoords="offset points", zorder=10)

    # 2. Add Logo
    try:
        logo_img = Image.open(logo_path)
    except (FileNotFoundError, PIL.UnidentifiedImageError):
        logging.warning("Error loading logo. Skipping logo overlay.")
        return  # Exit function if logo cannot be loaded

    logo_img.thumbnail((25, 25))  # Resize logo to 25x25 pixels

    # Calculate logo position with margin
    dpi = 100
    margin_pixels = int(1 * dpi)  # 1 cm margin
    x_pos = ax.get_xlim()[1] - logo_img.width - margin_pixels
    y_pos = ax.get_ylim()[0] + margin_pixels

    logo = OffsetImage(logo_img)
    ab_logo = AnnotationBbox(logo, (x_pos, y_pos),
                            frameon=False,
                            xycoords='data',
                            boxcoords="offset points",
                            box_alignment=(1, 0),
                            zorder=10)
    ax.add_artist(ab_logo)




# --- Placeholder Commands (For Future Implementation) ---

@bot.command()
async def alerts(ctx, location: str = None):
    """Fetches and displays current weather alerts for a specified location or the user's location."""

    await ctx.send("This feature is not yet implemented. Stay tuned for updates!")

    # TODO:
    # 1. If no location, use user's location from profile (if available) or prompt for input.
    # 2. Fetch alerts from NWS API (https://www.weather.gov/documentation/services-web-api)
    # 3. Filter alerts by type and/or severity (e.g., tornado, severe thunderstorm)
    # 4. Format alerts into a user-friendly message (embed is recommended)
    # 5. Send the message to the channel

@bot.command()
async def model(ctx, model_name: str, parameter: str, location: str):
    """Fetches and displays model forecast data for the specified parameter and location."""

    await ctx.send("This feature is not yet implemented. Stay tuned for updates!")

    # TODO:
    # 1. Validate model_name (GFS, NAM, HRRR, etc.) and parameter (temperature, wind, etc.)
    # 2. Fetch model data from a suitable source (NOMADS, Unidata, etc.)
    # 3. Parse GRIB2 data using libraries like xarray or metpy
    # 4. Extract the desired parameter for the specified location and time range
    # 5. Visualize the data (plot, table, or text summary)
    # 6. Send the formatted data to the channel

@bot.command()
async def lightning(ctx, region: str = None):
    """Displays lightning strike data for a specified region."""

    await ctx.send("This feature is not yet implemented. Stay tuned for updates!")

    # TODO:
    # 1. If no region, use user's location or prompt for input
    # 2. Fetch lightning data from a provider (Earth Networks, Vaisala, NLDN)
    # 3. Filter strikes by region and time range (if applicable)
    # 4. Visualize lightning strikes on a map (using Cartopy or similar)
    # 5. Send the map image or a text summary to the channel

@bot.command()
async def webcam(ctx, location: str):
    """Displays a weather webcam image for a specified location."""

    await ctx.send("This feature is not yet implemented. Stay tuned for updates!")

    # TODO:
    # 1. Find a suitable source for weather webcam images (many options exist)
    # 2. Fetch the image URL for the specified location or camera
    # 3. Send the image to the channel

# --- RAP Mesoscale Analysis College of DuPage ---
#@bot.command()
#async def cod(ctx, product_code: str):
#   """Fetches weather data from College of DuPage.
#   try:
#       product_info = config.RAP_PRODUCTS.get(product_code)
#        if not product_info:
#            raise ValueError("Invalid product code. Available options are: " + ", ".join(config.RAP_PRODUCTS.keys()))
#
#        image_url = product_info["url"]
#        response = requests.get(image_url)
#        response.raise_for_status()
#        img = Image.open(BytesIO(response.content))

        # (Optional) Add logo or other overlays to the image
        # ...

#        buffer = BytesIO()
#        img.save(buffer, format="PNG")
#        buffer.seek(0)

#        await ctx.send(file=discord.File(buffer, f"cod_{product_code}.png"))

#   except (requests.exceptions.RequestException, ValueError) as e:
#        await ctx.send(f"Error fetching COD data: {e}")
#this isn't finished yet
# trying to expand this still, we need more maps. ohh. probably won't do it. i might to the uhh unit conversion too. i know what to do to get the RAP mesoscale section to work, I just wanna go ahead and see if the both works and worry about this later

# i did have a help section that belongs here but it was giving me problems so i deleted it, not that important
