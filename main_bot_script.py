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
from math import radians, cos, sin, asin, sqrt, tan, csc, sec, cot

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
"""Fetches METARs for the specified airport code."""
def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)

        # Check if any METAR data was found at all
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        # Extract the raw METAR observation
        raw_metar = json_data[0]['rawText']  # Use 'rawText' instead of 'rawOb'

        if not raw_metar:
            raise ValueError("METAR data not found.")

        return raw_metar

    except requests.exceptions.RequestException as e:
        # Handle network errors during fetching
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        # Handle potential parsing errors
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

# rewrote both TAF and METAR sections to reflect the correct API pulling wizardry and the same error handling for uniformity

# --- TAF Command ---
@bot.command()
async def taf(ctx, airport_code: str):
    """Fetches TAF for the specified airport code."""
    try:
        # 1. Input Handling
        airport_code = airport_code.upper()

        # 2. Construct URL (adjust based on ADDS API changes)
        taf_url = f'https://aviationweather.gov/api/data/taf?ids={airport_code}&format=json'

        # 3. Fetch Data
        response = requests.get(taf_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # 4. Parse TAF
        json_data = json.loads(response.content)

        # Check if any TAF data was found at all
        if not json_data:
            raise ValueError(f"No TAF data found for {airport_code}.")

        # Extract the latest TAF (assuming the first one is the most recent)
        taf_data = json_data[0]['rawText']
        taf_time = json_data[0]['issueTime']

        if not taf_data or not taf_time:
            raise ValueError("TAF data or issue time not found.")

        # 5. Prepare and Send Response
        embed = discord.Embed(title=f"TAF for {airport_code}", description=taf_data)
        embed.set_footer(text=f"Issue Time: {taf_time}Z")
        await ctx.send(embed=embed)

        logging.info(f"User {ctx.author} requested TAF for {airport_code}")

    # 6. Error Handling 
    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching TAF data for {airport_code}: {e}")
        logging.error(f"Error fetching TAF data for {airport_code}: {e}")
    except (KeyError, ValueError) as e:  # Handle potential parsing errors
        await ctx.send(f"Error parsing TAF data for {airport_code}: {e}")
        logging.error(f"Error parsing TAF data for {airport_code}: {e}")

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

        # Save the Skew-T diagram temporarily
        temp_image_path = f"skewt_{station_code}_observed.png"
        plt.savefig(temp_image_path, format='png')
        plt.close(fig)

        # Add the bot avatar overlay
        add_bot_avatar_overlay(None, temp_image_path, avatar_url="https://your-bot-avatar-url.jpg", logo_size=50)

        # Send the stamped image as a Discord file
        await ctx.send(file=discord.File(temp_image_path))

        # Clean up the temporary image file
        os.remove(temp_image_path)

    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing Skew-T data for {station_code}: {e}")
	    
# --- Satellite Command ---
@bot.command()
async def sat(ctx, region: str, product_code: int):
    """Fetches satellite image for the specified region and product code using pre-defined links."""

    try:
        region = region.lower()
        valid_regions = ["conus", "fulldisk", "mesosector1", "mesosector2", "tropicalatlantic", "gomex", "ne", "sp", "mw", "nw", "sw", "pac"]

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")

        # Product codes for different regions
        product_codes = {
            "conus": {1: "GeoColor (True Color)", 2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "fulldisk": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "mesosector1": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "mesosector2": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "tropicalatlantic": {1: "GeoColor (True Color)", 2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "gomex": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "ne": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "sp": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"}, 
            "mw": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"}, 
            "nw": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"}, 
            "sw": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"}, 
            "pac": {9: "Mid-level Water Vapor", 14: "Clean Longwave Infrared Window", 22: "RGB"} 
        }

        # Error handling for invalid product code
        if product_code not in product_codes[region]:
            raise ValueError(f"Invalid product code for {region}. Valid codes are: {', '.join(map(str, product_codes[region].keys()))}")

        # Define base URLs for different GOES satellites and regions
        base_urls = {
            "goes16": {
                "conus": "https://whirlwind.aos.wisc.edu/~wxp/goes16/",
                "fulldisk": "https://whirlwind.aos.wisc.edu/~wxp/goes16/",
                "mesosector1": "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/",
                "mesosector2": "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/",
                "tropicalatlantic": "https://whirlwind.aos.wisc.edu/~wxp/goes16/",
                "gomex": "https://whirlwind.aos.wisc.edu/~wxp/goes16/",
                "ne": "https://whirlwind.aos.wisc.edu/~wxp/goes16/",
                "sp": "https://whirlwind.aos.wisc.edu/~wxp/goes16/",
                "mw": "https://whirlwind.aos.wisc.edu/~wxp/goes16/"
            },
            "goes17": {
                "nw": "https://whirlwind.aos.wisc.edu/~wxp/goes17/",
                "sw": "https://whirlwind.aos.wisc.edu/~wxp/goes17/",
                "pac": "https://whirlwind.aos.wisc.edu/~wxp/goes17/"
            }
        }

        # Define product paths based on product code
        product_paths = {
            1: {  # GeoColor (True Color)
                "conus": "geocolor/conus/",
                "fulldisk": "geocolor/fulldisk_full/",
                "mesosector1": "geocolor/meso_1/",
                "mesosector2": "geocolor/meso_2/",
                "tropicalatlantic": "geocolor/tropical_atlantic/"
            },
            2: {  # Red Visible
                "conus": "vis/conus/",
                "tropicalatlantic": "vis/tropical_atlantic/",
                "mesosector2": "grb/meso_vis_sqrt/",
                "gomex": "vis/gulf/",
                "ne": "vis/ne/",
                "sp": "vis/sp/",
                "mw": "vis/mw/",
                "nw": "vis/nw/",
                "sw": "vis/sw/"
            },
            9: {  # Mid-level Water Vapor
                "conus": "wvc/conus/",
                "fulldisk": "wvc/fulldisk_full/",
                "mesosector1": "grb/meso_wvc/",
                "mesosector2": "grb/meso_wvc/",
                "tropicalatlantic": "wvc/tropical_atlantic/",
                "gomex": "wvc/gulf/",
                "ne": "wvc/ne/",
                "sp": "wvc/sp/",
                "mw": "wvc/mw/",
                "nw": "wvc/nw/",
                "sw": "wvc/sw/",
                "pac": "wvc/namer/"
            },
            13: {  # Clean Longwave Infrared Window (mesosector1, mesosector2)
                "mesosector1": "grb/meso_ircm/",
                "mesosector2": "grb/meso_ircm/"
            },
            14: {  # Clean Longwave Infrared Window (other regions)
                "conus": "ircm/conus/",
                "tropicalatlantic": "ircm/tropical_atlantic/",
                "fulldisk": "ircm/fulldisk_full/",
                "gomex": "ircm/gulf/",
                "ne": "ircm/ne/",
                "sp": "ircm/sp/",
                "mw": "ircm/mw/",
                "nw": "ircm/nw/",
                "sw": "ircm/sw/",
                "pac": "irc13m/namer/" 
            },
            22: {  # RGB
                "conus": "https://dustdevil.aos.wisc.edu/goes16/grb/rgb/conus/",
                "tropicalatlantic": "https://dustdevil.aos.wisc.edu/goes16/grb/rgb/tropical_atlantic/",
                "sw": "https://dustdevil.aos.wisc.edu/goes17/grb/rgb/sw/",
                "pac": "https://dustdevil.aos.wisc.edu/goes17/grb/rgb/namer/"
            }
        }

        # Get the image URL based on region and product code
        image_url = image_links.get((region, product_code))
        if not image_url:
            # If not found in image_links, construct the URL
            satellite = "goes16" if region in base_urls["goes"]
		
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
async def radar(ctx, region: str = "plains", overlay: str = "base"):
    """Displays a radar image for the specified region and overlay type."""

    try:
        region = region.lower()
        overlay = overlay.lower()

        valid_regions = ["plains", "ne", "se", "sw", "nw"]
        valid_overlays = ["base", "totals"]

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")
        if overlay not in valid_overlays:
            raise ValueError(f"Invalid overlay. Valid options are: {', '.join(valid_overlays)}")

        # Radar image links
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
        }

        # Get the image URL based on region and overlay
        image_url = image_links.get((region, overlay))
        if not image_url:
            raise ValueError("Invalid region/overlay combination.")

        # Fetch the image content
        response = requests.get(image_url)
        response.raise_for_status()

        # Save the image temporarily
        temp_image_path = "temp_radar_image.gif"
        with open(temp_image_path, "wb") as f:
            f.write(response.content)

        # Add the bot avatar overlay
        add_bot_avatar_overlay(None, temp_image_path, avatar_url="https://your-bot-avatar-url.jpg", logo_size=50)

        # Send the stamped image as a Discord file
        await ctx.send(file=discord.File(temp_image_path, filename="radar.gif"))

        # Clean up the temporary image file
        os.remove(temp_image_path)

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
# ens before this works, the file names and source maybe for it? needs to be correct. this was just some generic code written with "logo.png". I dont know what the bots avatar's name is and if this just isn't possible or its just too silly, then I'll revert the code. I just thought you'd be able to make it work from here 

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
        active_storms = extract_active_storms(soup)

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
        image_urls = extract_image_urls(soup)

        # Download and send images
        for image_url in image_urls:
            image_filename = image_url.split('/')[-1]
            urllib.request.urlretrieve(image_url, image_filename)
            await ctx.send(file=discord.File(image_filename))

    except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing ASCAT imagery: {e}")

# Functions for parsing 
def extract_active_storms(soup):
    """Parses the BeautifulSoup object (soup) to extract a list of active storm IDs."""
    active_storms_table = soup.find('table', {'id': 'active_storm_table'})
    if active_storms_table:
        storm_links = active_storms_table.find_all('a')
        return [link.text.strip() for link in storm_links]
    else:
        return []  # No active storms found

def extract_image_urls(soup):
    """Parses the BeautifulSoup object (soup) to extract a list of image URLs."""
    image_tags = soup.find_all('img', {'class': 'product_image'})
    base_url = "https://www.fnmoc.navy.mil" 
    return [base_url + img['src'] for img in image_tags]

# --- Alerts Command ---
@bot.command()
async def alerts(ctx, location: str = None):
    """Fetches and displays current weather alerts for a specified location or the user's location."""

    if location is None:
        # ... (same as before, handle user location if not provided)

    location = location.lower()  # Convert input to lowercase for easier comparison

    if location in state_abbreviations_to_fips:
        state_fips = state_abbreviations_to_fips[location]
        alerts_url = f"https://api.weather.gov/alerts/active?area={state_fips}" 
    else:
        # ... (handle other location types if needed, or provide an error message)
        await ctx.send("Invalid location. Please provide a two-letter state abbreviation (e.g., 'ga' for Georgia).")
        return

    response = requests.get(alerts_url)

    if response.status_code == 200:
        alerts_data = response.json()

        filtered_alerts = []
        for alert in alerts_data['features']:
            event = alert['properties']['event']
            severity = alert['properties']['severity']

            # Customize filtering criteria here if needed
            filtered_alerts.append(alert)

        if filtered_alerts:
            for alert in filtered_alerts:
                properties = alert['properties']
                embed = discord.Embed(
                    title=properties['headline'],
                    color=discord.Color.red() 
                )
                embed.add_field(name="Severity", value=properties['severity'], inline=True)
                embed.add_field(name="Effective", value=properties['onset'], inline=True)
                embed.add_field(name="Expires", value=properties['expires'], inline=True)
                embed.add_field(name="Area", value=", ".join(properties['areaDesc']), inline=False)
                embed.add_field(name="Description", value=properties['description'], inline=False)
                embed.add_field(name="Instructions", value=properties['instruction'] or "None", inline=False)
                await ctx.send(embed=embed)
        else:
            await ctx.send("No weather alerts found for the specified location.")

    else:
        await ctx.send(f"Error fetching alerts: {response.status_code}")

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

# need to implement this function to get airport coordinates so lightning command can work
def get_airport_coordinates(icao):
    # ... (future code here)

def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Earth's   
 radius in miles
    R = 3956 

    # Convert latitude and longitude to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1
    dlon = lon2   
 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Distance   
 in miles
    distance = R * c
    
    return distance

# --- Lightning Command ---
@bot.command()
async def lightning(ctx, icao: str, radius: int = 5):
    """Checks for lightning strikes within a specified radius of an ICAO airport."""

    try:
        # 1. Get airport coordinates (you'll need to implement this function)
        airport_coords = get_airport_coordinates(icao)
        if airport_coords is None:
            await ctx.send(f"Could not find airport with ICAO code {icao}.")
            return

        # 2. Construct the API URL (replace with your actual credentials)
        api_url = f'https://data.api.xweather.com/lightning/{airport_coords[0]},{airport_coords[1]}?format=json&filter=cg&limit=10&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET'

        # 3. Fetch lightning data
        request = urllib.request.urlopen(api_url)
        response = request.read()
        data = json.loads(response)
        request.close()

        # 4. Check for success and process data
        if data['success']:
            lightning_data = data['data']

            # 5. Check for strikes within the radius (using the provided distance function)
            has_lightning = any(
                distance(strike['latitude'], strike['longitude'], airport_coords[0], airport_coords[1]) <= radius
                for strike in lightning_data
            )

            # 6. Send the result
            if has_lightning:
                await ctx.send(f"Lightning detected within {radius} miles of {icao}.")
            else:
                await ctx.send(f"No lightning detected within {radius} miles of {icao}.")

        else:
            await ctx.send(f"An error occurred: {data['error']['description']}")

    except Exception as e:
        await ctx.send(f"Error checking for lightning: {e}")

# --- Webcam Command --- 
@bot.command()
async def webcam(ctx, location: str):
    """Displays a weather webcam image for a specified location."""

    await ctx.send("This feature is not yet implemented. Stay tuned for updates!")

    # i got it written idk if it'll break the bot or not, im not comfortable with this command

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        filename="weather_bot.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    bot.run(config.DISCORD_TOKEN)
