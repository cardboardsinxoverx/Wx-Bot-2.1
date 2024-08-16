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
import cartopy
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
import json
# Load Configuration
import config
import signal


def save_cache(cache_type, data):
    with open(f"{cache_type}_cache.json", "w") as f:
        json.dump(data, f, indent=2)


def load_cache(cache_name):
    try:
        with open(f"{cache_name}_cache.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the cache file doesn't exist


# Data Storage (Caching)
metar_cache = {}
taf_cache = {}
alert_cache = {}

# Load Cache Data on Startup
metar_cache = load_cache("metar")  # keep getting and error with this line sometimes ohh
taf_cache = load_cache("taf") # guess maybe this one too
alert_cache = load_cache("alert") # this is gay

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}') # idk i hope this works
    # channel = client.get_channel(459016306960760834)
	# await channel.send(random.choice([
    #     "What's up bitches! I'm back!",
    #     "Hello again",
    #     "pls",
    #     "Let me guess, you want a METAR for kmge?",
    #     "We don't call 911",
    #     "Welcome to The Thunderdome!",
    #     "#hillarysemails"
    #     ]))


# --- on_message Event Handler ---
@bot.event
async def on_message(message):
    if message.author == bot.user:  # Don't respond to self
        return
    await bot.process_commands(message)  # Process bot commands


def get_metar(icao, hoursback=0, format='json'):
    metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
    src = requests.get(metar_url).content
    json_data = json.loads(src)
    print(json_data[0]['rawOb'])
    return json_data[0]['rawOb']

# --- Restart Command ---
@bot.command()
async def restart(ctx):
    """Restarts the bot."""
    try:
        await ctx.send("Restarting...")
        # Get the process ID of the current Python process
        pid = os.getpid()
        # Send SIGTERM signal to gracefully terminate the process
        os.kill(pid, signal.SIGTERM) 
    except Exception as e:
        await ctx.send(f"Error during restart: {e}")
# no idea of that works or not, lets find out

# --- METAR Command ---
@bot.command(aliases=["wx"])
async def metar(ctx, airport_code: str, hours_ago: int = config.DEFAULT_HOURS_BEFORE_NOW_METAR):
    """Fetches METAR for the specified airport code. Optionally specify hours ago for historical data."""
    try:
        if hours_ago < 0:
            raise ValueError("Invalid hours ago. Please enter a non-negative number.")

        # --- 1. Construct URL ---
        # url = f"{config.AVIATION_WEATHER_URL}?dataSource=metars&requestType=retrieve&format=xml&stationString={airport_code.upper()}&hoursBeforeNow={hours_ago}"  # Uppercase for consistency

        # # --- 2. Fetch Data ---
        # response = requests.get(url)
        # response.raise_for_status()  # Raise an exception if the request fails

        # # --- 3. Parse METAR ---
        # soup = BeautifulSoup(response.content, 'xml')
        # metar_data = soup.find('raw_text').text

        metar_data = get_metar(airport_code, hours_ago)
        if not metar_data:
            raise ValueError(f"METAR data not found for {airport_code}.")

        # --- 4. Extract Time (for Historical METARs) ---
        if hours_ago > 0:
            # metar_time = soup.find('observation_time').text
            metar_time = metar_data.split(' ')[1]
            message = f"METAR for {airport_code} ({metar_time}): {metar_data}"

        else:
            metar_time = None  # Current METAR doesn't need time in output
            message = f"METAR for {airport_code}: {metar_data}"

        # --- 5. Prepare and Send Response ---
        # if metar_time:
        # else:

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
async def taf(ctx, airport_code: str, *args):
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
async def skewt(ctx, station_code: str):
    """Fetches sounding data from the University of Wyoming and generates a Skew-T diagram."""

    try:
        station_code = station_code.upper()

        # Construct the URL for the Wyoming sounding archive
        sounding_url = f"https://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR=latest&MONTH=latest&FROM=0000&TO=2300&STNM={station_code}"

        # Fetch the sounding data
        response = requests.get(sounding_url)
        response.raise_for_status()

        # Parse the HTML to extract the sounding text
        soup = BeautifulSoup(response.content, 'html.parser')
        sounding_data = soup.find("pre").text.strip()

        if not sounding_data:
            raise ValueError("Sounding data not found.")

        # Generate the Skew-T diagram using SHARPpy
        profile = sharppy.Profile.from_sounding(sounding_data)
        fig = plt.figure(figsize=(8, 8))
        skew.plot(profile)

        # Prepare and send the image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        await ctx.send(file=discord.File(buffer, f"skewt_{station_code}_observed.png"))

    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing Skew-T data for {station_code}: {e}")

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
    """Provides sunrise, sunset, moon phase, and twilight information for a given location."""
    if not location:
        await ctx.send("Please provide a location (e.g., '$astro New York City' or '$astro kmge'") # i haven't found a way to specify cities like the 95 jacksonvilles in the USA, so its a lot easier to just type in the ICAO
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

        # Calculate sunrise, sunset, and twilight times using PyEphem
        obs = ephem.Observer()
        obs.lat = str(loc.latitude)
        obs.long = str(loc.longitude)
        obs.date = now
        sun = ephem.Sun()
        moon = ephem.Moon()

        sunrise = obs.next_rising(sun).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))
        sunset = obs.next_setting(sun).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))

        # Twilight calculations
        obs.horizon = '-0:34'  # Civil twilight
        civil_twilight_begin = obs.previous_rising(sun, use_center=True).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))
        civil_twilight_end = obs.next_setting(sun, use_center=True).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))

        obs.horizon = '-6'  # Civil twilight
        nautical_twilight_begin = obs.previous_rising(sun, use_center=True).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))
        nautical_twilight_end = obs.next_setting(sun, use_center=True).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))

        obs.horizon = '-12'  # Astronomical twilight
        astronomical_twilight_begin = obs.previous_rising(sun, use_center=True).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))
        astronomical_twilight_end = obs.next_setting(sun, use_center=True).datetime().replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))

        # Calculate moon phase
        moon.compute(now)
        moon_phase = moon.phase

        # Format the results
        response = f"Astronomy information for {location}:\n\n"
        response += f"**Sunrise:** {sunrise.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Sunset:** {sunset.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Civil Twilight Begin:** {civil_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Civil Twilight End:** {civil_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Nautical Twilight Begin:** {nautical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Nautical Twilight End:** {nautical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Astronomical Twilight Begin:** {astronomical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Astronomical Twilight End:** {astronomical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        response += f"**Moon Phase:** {moon_phase:.1f}% (Illuminated)"

        await ctx.send(response)
        logging.info(f"User {ctx.author} requested astronomy information for {location}")
    except (GeocoderTimedOut, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving astronomy information: {e}")
        logging.error(f"Error retrieving astronomy information for {location}: {e}")

# --- Radar Command ---
@bot.command()
async def radar(ctx, region: str = "plains", overlay: str = "base"):  # Changed default overlay to "base"
    """Displays a radar image for the specified region and overlay type."""

    try:
        region = region.lower()
        overlay = overlay.lower()

        valid_regions = ["plains", "ne", "se", "sw", "nw"] 
        valid_overlays = ["base", "totals"]  # Replaced "none" with "base"

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")
        if overlay not in valid_overlays:
            raise ValueError(f"Invalid overlay. Valid options are: {', '.join(valid_overlays)}")

        # Radar image links (updated with "base" instead of "none")
        image_links = {
            ("plains", "base"): "https://tempest.aos.wisc.edu/radar/plains3comp.gif",
            ("plains", "totals"): "https://tempest.aos.wisc.edu/radar/plainsPcomp.gif",
            ("ne", "base"): "https://tempest.aos.wisc.edu/radar/ne3comp.gif",
            ("ne", "totals"): "https://tempest.aos.wisc.edu/radar/nePcomp.gif",
            ("se", "base"): "https://tempest.aos.wisc.edu/radar/se3comp.gif",
            ("se", "totals"): "https://tempest.aos.wisc.edu/radar/sePcomp.gif",
            ("sw", "base"): "https://tempest.aos.wisc.edu/radar/sw3comp.gif",
            ("sw", "totals"): "https://tempest.aos.wisc.edu/radar/swPcomp.gif",
            ("nw", "base"): "https://tempest.aos.wisc.edu/radar/nw3comp.gif",
            ("nw", "totals"): "https://tempest.aos.wisc.edu/radar/nwPcomp.gif",
            # ... should be all the links we need for radar, like I said not looking for anything fancy here because obviously we use things like radarscope and grlevel3, but having a bot that can pull an image that's ready to save and post
	    # is better than having to go to this website and click through the 95 links, right click, possibly needing change the format. you get what i mean
        }

    except (requests.exceptions.RequestException, ValueError) as e:
        await ctx.send(f"Error retrieving radar image: {e}")

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




# --- Placeholder Commands (these are now UP and may or may not work) ---

# --- ASCAT Command ---
@bot.command()
async def ascat(ctx, storm_id: str = None):
    """Fetches ASCAT images for the specified storm from FNMOC. If no storm_id is provided, it will list the active storms."""

    try:
        # Fetch the main FNMOC TCWEB page
        base_url = "https://www.fnmoc.navy.mil/tcweb/cgi-bin/tc_home.cgi"
        response = requests.get(base_url)
        response.raise_for_status()

        # Parse the HTML to find active storms
        soup = BeautifulSoup(response.content, 'html.parser')
        active_storms = extract_active_storms(soup)  # Replace with your actual parsing logic

        if storm_id is None:
            # If no storm_id is provided, list the active storms
            if active_storms:
                await ctx.send(f"Currently active storms: {', '.join(active_storms)}")
            else:
                await ctx.send("No active storms found.")
            return  # Exit the command early

        # Check if the requested storm is active
        if storm_id.upper() not in [s.upper() for s in active_storms]:
            raise ValueError(f"Storm '{storm_id}' not found among active storms. Currently active storms are: {', '.join(active_storms)}")

        # Construct the URL for the storm's ASCAT image page (adjust as needed)
        storm_url = f"{base_url}?YEAR=2024&MO=Aug&BASIN=ATL&STORM_NAME={storm_id}&SENSOR=&PHOT=yes&ARCHIVE=Mosaic&NAV=tc&DISPLAY=all&MOSAIC_SCALE=20%&STYLE=table&ACTIVES={','.join(active_storms)}&TYPE=ascat&CURRENT=LATEST.jpg&PROD=hires&DIR=/tcweb/dynamic/products/tc24/ATL/{storm_id}/ascat/hires&file_cnt=160"

        # Fetch the storm's ASCAT image page
        response = requests.get(storm_url)
        response.raise_for_status()

        # Parse the HTML to extract image URLs
        soup = BeautifulSoup(response.content, 'html.parser')
        image_urls = extract_image_urls(soup)  # Replace with your actual parsing logic

        # Download and send images
        for image_url in image_urls:
            image_filename = image_url.split('/')[-1]
            urllib.request.urlretrieve(image_url, image_filename)
            await ctx.send(file=discord.File(image_filename))

    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing ASCAT imagery: {e}")

# Placeholder functions for parsing (you'll need to implement these)
def extract_active_storms(soup):
    """Parses the BeautifulSoup object (soup) to extract a list of active storm IDs."""
    # ... your implementation here
    pass

def extract_image_urls(soup):
    """Parses the BeautifulSoup object (soup) to extract a list of image URLs."""
    # ... your implementation here
    pass


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


# --- Models Command, under the command $weather ---
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)   


@bot.command()
async def weather(ctx, location: str, *, variables: str = None):
    """Fetches hourly weather data for a specified location from Open-Meteo."""

    try:
        # 1. Get coordinates for the location (you might need to implement this)
        latitude, longitude = get_coordinates(location)  # Implement this function

        # 2. Define the list of weather variables to fetch
        available_variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", 
            "apparent_temperature", "precipitation_probability", "precipitation", 
            "rain", "showers", "snowfall", "snow_depth", "weather_code", 
            "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", 
            "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", 
            "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", 
            "wind_speed_80m", "wind_direction_10m",   
 "wind_direction_80m", "wind_gusts_10m", 
            "temperature_80m", "surface_temperature", "soil_temperature_0_to_10cm", 
            "soil_temperature_10_to_40cm", "uv_index", "sunshine_duration", "cape", 
            "lifted_index", "convective_inhibition", "freezing_level_height", 
            "temperature_1000hPa", "temperature_925hPa", "temperature_850hPa", 
            "temperature_700hPa", "temperature_500hPa", "temperature_300hPa", 
            "dew_point_1000hPa", "dew_point_925hPa", "dew_point_800hPa", 
            "dew_point_700hPa", "dew_point_500hPa", "dew_point_300hPa", 
            "wind_speed_1000hPa", "wind_speed_925hPa", "wind_speed_850hPa", 
            "wind_speed_700hPa", "wind_speed_500hPa", "wind_speed_300hPa", 
            "wind_direction_1000hPa", "wind_direction_925hPa", "wind_direction_850hPa", 
            "wind_direction_700hPa", "wind_direction_500hPa", "wind_direction_300hPa", 
            "vertical_velocity_1000hPa", "vertical_velocity_925hPa", "vertical_velocity_850hPa", 
            "vertical_velocity_700hPa", "vertical_velocity_500hPa", "vertical_velocity_300hPa", 
            "geopotential_height_1000hPa", "geopotential_height_925hPa", "geopotential_height_850hPa", 
            "geopotential_height_700hPa", "geopotential_height_500hPa", "geopotential_height_300hPa"
        ]

        if variables:
            requested_variables = [var.strip() for var in variables.split(',')]
            # Check if requested variables are valid
            invalid_variables = [var for var in requested_variables if var not in available_variables]
            if invalid_variables:
                await ctx.send(f"Invalid variables: {', '.join(invalid_variables)}. Available variables: {', '.join(available_variables)}")
                return
        else:
            # Default to some basic variables if none are specified
            requested_variables = ["temperature_2m", "precipitation_probability", "wind_speed_10m"]

        # 3. Fetch weather data
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": requested_variables,
            "wind_speed_unit": "kn",
            "precipitation_unit": "inch",
            "timezone": "America/New_York",  # Adjust timezone if needed
            "past_hours": 6,
            "forecast_hours": 12,
            "models": ["gfs_seamless", "gfs_global", "gfs_hrrr", "gfs_graphcast025"]  # Adjust models if needed
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # 4. Process hourly data
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        for i, var_name in enumerate(requested_variables):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

        hourly_dataframe = pd.DataFrame(data=hourly_data)

        # 5. Create and send embedded message (you'll likely want to customize this further)
        embed = discord.Embed(title=f"Weather Forecast for {location}", color=discord.Color.blue())
        for var_name in requested_variables:
            embed.add_field(name=var_name, value=hourly_dataframe[var_name].to_string(index=False), inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"Error fetching weather data: {e}")

def get_coordinates(location):
    """Gets the latitude and longitude for a given location using Nominatim."""

    geolocator = Nominatim(user_agent="WeatherBot")  # Replace with your bot's name in "_"
    location_obj = geolocator.geocode(location)

    if location_obj:
        return location_obj.latitude, location_obj.longitude
    else:
        return None
# new

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

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        filename="weather_bot.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    bot.run(config.DISCORD_TOKEN)
