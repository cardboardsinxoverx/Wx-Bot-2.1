## botsucks

# Imports and Setup
import sys
import time
import subprocess
import random
#print(sys.path)
import discord
from discord.ext import commands
import requests
import urllib3
import openmeteo_py
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
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import geocoder
import json
import psutil
import config
import signal
import math
import metpy
import requests_cache
from openmeteo_py import Options,OWmanager
from retry import retry
import openmeteo_requests
from openmeteo_py import Hourly, Options, Variable
import airportsdata

# def save_cache(cache_type, data):
#     with open(f"{cache_type}_cache.json", "w") as f:
#         json.dump(data, f, indent=2)

# def load_cache(cache_name):
#     try:
#         with open(f"{cache_name}_cache.json", "r") as file:
#             return json.load(file)
#     except FileNotFoundError:
#         return {}  # Return an empty dictionary if the cache file doesn't exist

# # Data Storage (Caching)
# metar_cache = {}
# taf_cache = {}
# alert_cache = {}

# # Load Cache Data on Startup
# metar_cache = load_cache("metar")  
# taf_cache = load_cache("taf") 
# alert_cache = load_cache("alert") 
# ^^^ that stuff up there is really only relevant if we were to disseminate this to the public. Imagine if hundreds of discords used this bot, pulling API requests. I mean, its not OUR internet (i guess?) but.. i think for stuff like the model command, which is really data heavy, it might be
# a good idea to implement the cache function back into the bot. So I wouldn't call it errenous AI bullshit, it just doesn't fit our needs at the moment. It's a lovely idea however.

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    #await ctx.send("What up bitches")
    # idk i hope this works
    channel = bot.get_channel(459016306960760834)
    await channel.send(random.choice([
        "What's up bitches! I'm back!",
        "Hello again",
        "pls",
        "Let me guess, you want a METAR for kmge?",
        "We don't call 911",
        "Welcome to The Thunderdome!",
        "#hillarysemails"
        ]))

# --- on_message Event Handler ---
@bot.event
async def on_message(message):
    channel = message.channel
    if message.author == bot.user:  # Don't respond to self
        return
    if message.content.startswith('bad bot'):
        await channel.send(random.choice([
        "No u",
        "Do you want to do this for me??",
        "meep.",
        "hua",
        "get rekt",
        "no setp on snk"
        ]))
    await bot.process_commands(message)  # Process bot commands

# --- Maps Command --- 
@bot.command()
async def map(ctx):
    """Sends a map image."""

    try:
        image_url = "https://github.com/cardboardsinxoverx/Wx-Bot-2.1/blob/main/sondes.jpg"  # please dont replace 

        # Fetch the image content
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the image temporarily
        temp_image_path = "temp_map_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(response.content)

        # Send the image as a Discord file
        await ctx.send(file=discord.File(temp_image_path))

        # Clean up the temporary image file
        os.remove(temp_image_path)

    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching map image: {e}")
	    
# --- git pull command ---
@bot.command()
async def pull(ctx):
    await ctx.send("Pulling down changes from GitHub...")
    res = subprocess.run(['git', 'pull'], capture_output=True, text=True)
    await ctx.send(f'{res}\nFinished. Good luck.')

# --- Restart Command ---
@bot.command()
async def restart(ctx):
    """Restarts the bot."""
    try:
        await ctx.send("Restarting...")
        # Get the process ID of the current Python process
        #pid = os.getpid()
        # Send SIGTERM signal to gracefully terminate the process
        #os.kill(pid, signal.SIGTERM) 
        #time.sleep(0.5)
        #python = sys.executable
        #os.execl(python, python, *sys.argv)
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            os.close(handler.fd)
        time.sleep(0.5)
        python = sys.executable
        os.execl(python, python, *sys.argv)
        await ctx.send("Back online bitches")
    except Exception as e:
        await ctx.send(f"Error during restart: {e}")


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
        raw_metar = json_data[0]['rawOb']  # Use 'rawText' instead of 'rawOb'

        if not raw_metar:
            raise ValueError("METAR data not found.")

        return raw_metar

    except requests.exceptions.RequestException as e:
        # Handle network errors during fetching
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        # Handle potential parsing errors
        raise Exception(f"Error parsing METAR data for {icao}: {e}")


@bot.command()
async def metar(ctx, airport_code: str):
    """Sends latest metar for given airport"""
    airport_code = airport_code.upper()
    raw_metar = get_metar(airport_code)
    
    embed = discord.Embed(title=f"METAR for {airport_code}", description=raw_metar)
    await ctx.send(embed=embed)
    logging.info(f"User {ctx.author} requested METAR for {airport_code}")


# --- TAF Command ---
@bot.command()
async def taf(ctx, airport_code: str):
    """Fetches TAF for the specified airport code."""
    try:
        # 1. Input Handling
        airport_code = airport_code.upper()

        # 2. Construct URL (adjust based on ADDS API changes)
        taf_url = f'https://aviationweather.gov/api/data/taf?ids={airport_code}&format=json'
	logging.info(f"Constructed TAF URL: {taf_url}")


        # 3. Fetch Data
        response = requests.get(taf_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
	logging.info(f"Fetched TAF data for {airport_code}")

        # 4. Parse TAF
        json_data = json.loads(response.content)
	logging.info(f"Parsed TAF JSON data for {airport_code}")

        # Check if any TAF data was found at all
        if not json_data:
            raise ValueError(f"No TAF data found for {airport_code}.")

        # Extract the latest TAF (assuming the first one is the most recent)
        taf_data = json_data[0]['rawText']
        taf_time = json_data[0]['issueTime']

        if not taf_data or not taf_time:
            raise ValueError("TAF data or issue time not found.")

       # Extract the latest TAF 
        if 'rawText' in json_data[0] and 'issueTime' in json_data[0]:
            taf_data = json_data[0]['rawText']
            taf_time = json_data[0]['issueTime']
        else:
            raise KeyError("TAF data or issue time not found in the response.")

        # 5. Prepare and Send Response
        embed = discord.Embed(title=f"TAF for {airport_code}", description=taf_data)
        embed.set_footer(text=f"Issue Time: {taf_time}Z")
        await ctx.send(embed=embed)

        logging.info(f"User {ctx.author} requested TAF for {airport_code}")

    # 6. Error Handling 
    except requests.exceptions.RequestException as e:
        # Handle network errors during fetching, including ConnectionError
        if isinstance(e, requests.exceptions.ConnectionError) and "Failed to resolve" in str(e):
            await ctx.send(f"Unable to connect to aviationweather.gov. Check your internet connection or DNS settings.")
        else:
            await ctx.send(f"Error fetching TAF data for {airport_code}: {e}")
        logging.error(f"Error fetching TAF data for {airport_code}: {e}")
    except (KeyError, ValueError) as e: 
        await ctx.send(f"Error parsing TAF data for {airport_code}: {e}")
        logging.error(f"Error parsing TAF data for {airport_code}: {e}")

taf.help = """
**$taf <airport_code>**

Fetches the latest TAF (terminal aerodrome forecast) for the specified airport.

**Arguments:**

*   `airport_code`: The ICAO code of the airport (e.g., 'KJFK', 'KATL').
"""

# more error handling fixes and put in logging info, also edited code to see if json data even exists before accessing it, but I'm tired of trying to fix the actual problem and the error handling problem at the same time.
# i think simply fixing the error handling after this try is the best way forward, at least then it can tell you whats wrong. so yeah last stab at fixing the actual problem in TAF command - ln

# --- SkewT Command --- 
@bot.command()
async def skewt(ctx, station_code: str):
    """Fetches sounding data and generates a Skew-T diagram with various indices."""

    try:
        station_code = station_code.upper()

	# Fetch and process the sounding data
        ds = xr.Dataset.from_dataframe(WyomingUpperAir.request_data(format_date, station.strip('K')))

        # Get today's date in YYYY-MM-DD format
        today = datetime.date.today().strftime("%Y-%m-%d")

        # Construct the URL 
        sounding_url = f"https://weather.uwyo.edu/cgi-bin/sounding?region=naconf&STNM={station_code}&DATE={today}&HOUR=latest&ENV=std"

        # Fetch the sounding data
        response = requests.get(sounding_url, verify=False)
        response.raise_for_status()

        # Parse the HTML to extract the sounding text
        soup = BeautifulSoup(response.content, 'html.parser')
        sounding_data = soup.find("pre").text.strip()

        if not sounding_data:
            raise ValueError("Sounding data not found from this WMO. This is likely because the sounding balloon was not released. Please check a neighboring WMO or try again later.")

        # Generate the Skew-T diagram using SHARPpy
        profile = sharppy.Profile.from_sounding(sounding_data)

        # Calculate indices 
        cape, cin = mpcalc.cape_cin(profile)
        lcl_pressure, lcl_temperature = mpcalc.lcl(profile.pres[0], profile.tmpc[0], profile.dwpc[0])
        lfc_pressure, lfc_temperature = mpcalc.lfc(profile.pres, profile.tmpc, profile.dwpc)
        el_pressure, el_temperature = mpcalc.el(profile.pres, profile.tmpc, profile.dwpc)
        lifted_index = mpcalc.lifted_index(profile.pres, profile.tmpc, profile.dwpc)
        pwat = mpcalc.precipitable_water(profile.pres, profile.dwpc)
        ccl_pressure, ccl_temperature = mpcalc.ccl(profile.pres, profile.tmpc, profile.dwpc)
        ehi = mpcalc.energy_helicity_index(profile.pres, profile.u, profile.v, profile.hght)
        zero_c_level = mpcalc.find_intersections(profile.pres, profile.tmpc, 0 * units.degC)[0]
        theta_e = mpcalc.equivalent_potential_temperature(profile.pres, profile.tmpc, profile.dwpc)
        k_index = mpcalc.k_index(profile.pres, profile.tmpc, profile.dwpc)
        mpl_pressure, mpl_temperature = mpcalc.mpl(profile.pres, profile.tmpc, profile.dwpc)
        max_temp = mpcalc.max_temperature(profile.pres, profile.tmpc, profile.dwpc)
        positive_shear = mpcalc.bulk_shear(p, u, v, height=slice(0, 3000 * units.m))
        srh = mpcalc.storm_relative_helicity(u, v, height, profile.storm_motion)
        total_totals = mpcalc.total_totals_index(profile.tmpc, profile.dwpc, profile.u, profile.v)
        # Assuming you have a function to calculate tropopause level 
        tropopause_level = calculate_tropopause_level(profile)

	# Calculate tropopause level using the birner function
        tropopause_level = birner(profile.pres, profile.tmpc, height=True)

        # Calculate wet-bulb temperature
        wet_bulb = mpcalc.wet_bulb_temperature(profile.pres, profile.tmpc, profile.dwpc)

        # Create the Skew-T plot
        fig = plt.figure(figsize=(8, 8))
        skew = SkewT(fig)

        # Pls plot
        skew.plot(p, T, 'r')  # Temperature in red
        skew.plot(p, Td, 'g')  # Dewpoint in green
        skew.plot(p, wet_bulb, 'b', linestyle='--')  # Wet-bulb temperature in dashed blue
        skew.plot_barbs(p[ix], u[ix], v[ix])  # Wind barbs 

        # mush indices on skewT
        plt.title(f'{station} {today} {profile.time[0].hour:02d}Z', weight='bold', size=20)
        skew.ax.text(0.7, 0.1, f'CAPE: {cape.to("J/kg"):.0f}', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.05, f'CIN: {cin.to("J/kg"):.0f}', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.15, f'LIFTED INDEX: {lifted_index:.0f}', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.2, f'LCL: {lcl_pressure.to("hPa").m:.0f} hPa', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.25, f'LFC: {lfc_pressure.to("hPa").m:.0f} hPa', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.3, f'EL: {el_pressure.to("hPa").m:.0f} hPa', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.35, f'CCL: {ccl_pressure.to("hPa").m:.0f} hPa', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.4, f'EHI: {ehi:.0f}', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.45, f'0C Level: {zero_c_level[0].to("hPa"):.0f} hPa, {zero_c_level[1].to("degC"):.0f} °C', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.5, f'PWAT: {pwat.to("inch"):.2f} in', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.55, f'K index: {k_index:.0f}', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.6, f'MPL: {mpl_pressure.to("hPa").m:.0f} hPa', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.65, f'Max Temp: {max_temp.to("degC"):.0f} °C', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.7, f'0-3km Shear: {positive_shear[0].to("knots"):.0f} kts', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.75, f'SRH: {srh[0].to("m^2/s^2"):.0f} m^2/s^2', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.8, f'Total Totals: {total_totals:.0f}', transform=skew.ax.transAxes)
        skew.ax.text(0.7, 0.85, f'Tropopause: {tropopause_level:.1f} km', transform=skew.ax.transAxes) 
	
	# set units for variables
	height = ds.height * units.meter

	# add hodograph
        ax_hod = inset_axes(skew.ax, '25%', '20%', loc='upper left')
        h = Hodograph(ax_hod, component_range=80)  # Change range in windspeeds
        h.add_grid(increment=10)
        try:
            h.plot_colormapped(u, v, height)
 	 
        except ValueError as e:
            print(e) 

        # Save the Skew-T diagram temporarily
        temp_image_path = f"skewt_{station_code}_observed.png"
        plt.savefig(temp_image_path, format='png')
        plt.close(fig)

        # Send the Skew-T diagram as a Discord file
        await ctx.send(file=discord.File(temp_image_path))

        # delete the temporary image file
        os.remove(temp_image_path)

	# error handling
   except (requests.exceptions.RequestException, AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving/parsing Skew-T data for {station_code}: {e}. This could be happening for several reasons, such as network connection issues, timeout errors, data not being in the correct format, or the bot is requesting data it wasn't programmed to understand.")
	    
# --- Satellite Command ---
@bot.command()
async def sat(ctx, region: str, product_code: int):
    """Fetches satellite image for the specified region and product code using pre-defined links."""

    try:
        region = region.lower()
        valid_regions = ["conus", "fulldisk", "mesosector1", "mesosector2", "tropicalatlantic", "gomex", "ne", "indian", "capeverde", "neatl", 'fl', 'pacus', 'wc', 'ak', 'wmesosector', 'wmesosector2'] 

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")

        # Product codes for different regions (updated with new regions and product codes)
        product_codes = {
            "conus": {1: "GeoColor (True Color)", 2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "fulldisk": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 5: "RGB Air Mass"},  # Changed "airmass" to 5
            "mesosector1": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "mesosector2": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "tropicalatlantic": {1: "GeoColor (True Color)", 2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "gomex": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "ne": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "fl": {2: "Red Visible"},
            "pacus": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "wc": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "ak": {9: "Mid-level Water Vapor", 14: "Clean Longwave Infrared Window", 22: "RGB"},
            "wmesosector": {2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "wmesosector2": {2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "indian": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "capeverde": {2: "Red Visible", 14: "Clean Longwave Infrared Window"},
            "neatl": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"}
        }

        # Error handling for invalid product code (simplified)
        if product_code not in product_codes[region]:
            raise ValueError(f"Invalid product code for {region}. Valid codes are: {', '.join(map(str, product_codes[region].keys()))}")

        # Define image_links with the provided URLs (organized by satellite and region)
        image_links = {
            # "eumetsat": {  # New section for EUMETSAT links
                "indian": {
                    14: "https://www.ssd.noaa.gov/eumet/indiano/rb.jpg",
                    2: "https://www.ssd.noaa.gov/eumet/indiano/vis.jpg",
                    9: "https://www.ssd.noaa.gov/eumet/indiano/wv.jpg"
                },
                "capeverde": {
                    14: "https://www.ssd.noaa.gov/eumet/eatl/rb.jpg",
                    2: "https://www.ssd.noaa.gov/eumet/eatl/vis.jpg"
                },
                "neatl": {
                    14: "https://www.ssd.noaa.gov/eumet/neatl/rb.jpg",
                    2: "https://www.ssd.noaa.gov/eumet/neatl/vis.jpg",
                    9: "https://www.ssd.noaa.gov/eumet/neatl/wv.jpg"
                },
            # }, # close eumetsat dictionary
            # "goes16": {
                "conus": {
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes16/ircm/conus/latest_conus_1.jpg",
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/vis/conus/latest_conus_1.jpg",
                    22: "https://dustdevil.aos.wisc.edu/goes16/grb/rgb/conus/latest_conus_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/wvc/conus/latest_conus_1.jpg"
                },
                "fulldisk": {
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes16/irc13m/fulldisk/latest_fulldisk_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/wvc/fulldisk/latest_fulldisk_1.jpg",
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/vis/fulldisk/latest_fulldisk_1.jpg",
                    "airmass": "https://whirlwind.aos.wisc.edu/~wxp/goes16/multi_air_mass_rgb/fulldisk/latest_fulldisk_1.jpg"
                },
                "mesosector1": {
                    13: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_ircm/latest_meso_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_wvc/latest_meso_1.jpg"
                },
                "mesosector2": {
                    13: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_ircm/latest_meso_2.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_wvc/latest_meso_2.jpg",
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_vis_sqrt/latest_meso_2.jpg"
                },
                "tropicalatlantic": {
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes16/ircm/tropical_atlantic/latest_tropical_atlantic_1.jpg",
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/vis/tropical_atlantic/latest_tropical_atlantic_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/wvc/tropical_atlantic/latest_tropical_atlantic_1.jpg",
                    22: "https://dustdevil.aos.wisc.edu/goes16/grb/rgb/tropical_atlantic/latest_tropical_atlantic_1.jpg"
                },
                "gomex": {
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes16/ircm/gulf/latest_gulf_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/wvc/gulf/latest_gulf_1.jpg",
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/vis/gulf/latest_gulf_1.jpg"
                },
                "ne": {
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/vis/ne/latest_ne_1.jpg",
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes16/ircm/ne/latest_ne_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/wvc/ne/latest_ne_1.jpg"
                },
                "fl": {
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/vis/fl/latest_fl_1.jpg"
                },
            # },  # close goes 16 dictionary
            # "goes17": {
                "pacus": {
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes17/vis/conus/latest_conus_1.jpg",
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes17/ircm/conus/latest_conus_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes17/wvc/conus/latest_conus_1.jpg",
                    22: "https://dustdevil.aos.wisc.edu/goes17/grb/rgb/conus/latest_conus_1.jpg"
                },
                "wc": {
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes17/vis/westcoast/latest_westcoast_1.jpg",
                    22: "https://dustdevil.aos.wisc.edu/goes17/grb/rgb/westcoast/latest_westcoast_1.jpg",
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes17/ircm/westcoast/latest_westcoast_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes17/wvc/westcoast/latest_westcoast_1.jpg"
                },
                "ak": {
                    22: "https://dustdevil.aos.wisc.edu/goes17/grb/rgb/ak/latest_ak_1.jpg",
                    14: "https://whirlwind.aos.wisc.edu/~wxp/goes17/irc13m/ak/latest_ak_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes17/wvc/ak/latest_ak_1.jpg"
                },
                "wmesosector2": {
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes17/grb/meso_vis/latest_meso_2.jpg",
                    13: "https://whirlwind.aos.wisc.edu/~wxp/goes17/grb/meso_ircm/latest_meso_2.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes17/grb/meso_wvc/latest_meso_2.jpg"
                },
                "wmesosector": {
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes17/grb/meso_vis/latest_meso_1.jpg",
                    13: "https://whirlwind.aos.wisc.edu/~wxp/goes17/grb/meso_ircm/latest_meso_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes17/grb/meso_wvc/latest_meso_1.jpg"
                } 
            # }  # Close goes17 dictionary
        }  # Close the main image_links dictionary

        # Retrieve the image URL
        image_url = image_links[region][product_code]

        if image_url:
            print(f"{product_codes[region][product_code]} for {region}:\n{image_url}")
            # Send the image as a Discord file
            # Fetch the image content
            response = requests.get(image_url)
            response.raise_for_status()

            # Save the image temporarily (use .jpg extension)
            temp_image_path = "temp_sat_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(response.content)

            # Send the image as a Discord file
            await ctx.send(file=discord.File(temp_image_path, filename="sat.jpg"))

            # Clean up the temporary image file
            os.remove(temp_image_path)

        else:
            raise KeyError(f"No image link found for region '{region}' and product code {product_code}")

    except (requests.exceptions.RequestException, AttributeError, ValueError, KeyError, OSError) as e:
        await ctx.send(f"Error retrieving/parsing/sending satellite imagery: {e}")

sat.help = """
**$sat <region> <product_code>**

Fetches a satellite image for the specified region and product code.

**Arguments:**

*   `region`: The geographic region. Valid options are:

    *   `conus`: Continental US (GOES-16)
    *   `fulldisk`: Full Earth disk (GOES-16)
    *   `mesosector1`, `mesosector2`: GOES-16 mesoscale sectors
    *   `tropicalatlantic`: Tropical Atlantic region (GOES-16)
    *   `gomex`: Gulf of Mexico (GOES-16)
    *   `ne`: Northeast US (GOES-16)
    *   `indian`: Indian Ocean (EUMETSAT)
    *   `capeverde`: Cape Verde region (EUMETSAT)
    *   `neatl`: Northeast Atlantic (EUMETSAT)
    *   `fl`: Florida (GOES-16)
    *   `pacus`: Pacific US (GOES-17)
    *   `wc`: West Coast US (GOES-17)
    *   `ak`: Alaska (GOES-17)
    *   `wmesosector`, `wmesosector2`: GOES-17 mesoscale sectors

*   `product_code`: The code representing the desired satellite product.

    **Available product codes for each region:**

    *   **CONUS:**
        *   `1`: GeoColor (True Color)
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `14`: Clean Longwave Infrared Window
        *   `22`: RGB

    *   **Full Disk:**
        *   `1`: GeoColor (True Color)
        *   `2`: Red Visible
        *   `5`: RGB Air Mass
        *   `9`: Mid-level Water Vapor
        *   `13`: Clean Longwave Infrared Window

    *   **Mesosector1 & Mesosector2:**
        *   `1`: GeoColor (True Color)
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `13`: Clean Longwave Infrared Window

    *   **Tropical Atlantic:**
        *   `1`: GeoColor (True Color)
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `14`: Clean Longwave Infrared Window
        *   `22`: RGB

    *   **GOMEX & NE:**
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `14`: Clean Longwave Infrared Window

    *   **FL:**
        *   `2`: Red Visible

    *   **PACUS, WC, & AK:**
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `14`: Clean Longwave Infrared Window
        *   `22`: RGB

    *   **WMesosector & WMesosector2:**
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `13`: Clean Longwave Infrared Window

    *   **Indian:**
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `14`: Clean Longwave Infrared Window

    *   **Cape Verde:**
        *   `2`: Red Visible
        *   `14`: Clean Longwave Infrared Window

    *   **NEATL:**
        *   `2`: Red Visible
        *   `9`: Mid-level Water Vapor
        *   `14`: Clean Longwave Infrared Window

**Example:**

*   `$sat conus 14` (Clean Longwave Infrared Window for CONUS)
"""
# updated links and dictionary format 18AUG2024
		
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
    # except (GeocoderTimedOut, AttributeError, ValueError) as e:
    except (AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving astronomy information: {e}")
        logging.error(f"Error retrieving astronomy information for {location}: {e}")

astro.help = """
**$astro [location]**

Provides sunrise, sunset, moon phase, and twilight information for a given location.

**Arguments:**

*   `location` (optional): The location for which you want to retrieve astronomy information.  You can provide a city name, airport code (ICAO), or other recognizable location. 
"""

# --- Radar Command ---
@bot.command()
async def radar(ctx, region: str = "chase", overlay: str = "base"):
    """Displays a radar image for the specified region and overlay type."""

    try:
        region = region.lower()
        overlay = overlay.lower()

        valid_regions = ["chase", "ne", "se", "sw", "nw"]
        valid_overlays = ["base", "totals"]

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")
        if overlay not in valid_overlays:
            raise ValueError(f"Invalid overlay. Valid options are: {', '.join(valid_overlays)}")

        # Radar image links
        image_links = {
            ("chase", "base"): "https://tempest.aos.wisc.edu/radar/chase3comp.gif",
            ("chase", "totals"): "https://tempest.aos.wisc.edu/radar/chasePcomp.gif",
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
        # add_bot_avatar_overlay(None, temp_image_path, avatar_url="https://your-bot-avatar-url.jpg", logo_size=50)

        # Send the stamped image as a Discord file
        await ctx.send(file=discord.File(temp_image_path, filename="radar.gif"))

        # Clean up the temporary image file
        os.remove(temp_image_path)

    except (requests.exceptions.RequestException, ValueError) as e:
        await ctx.send(f"Error retrieving radar image: {e}")

radar.help = """
**$radar [region] [overlay]**

Displays a radar image link for the specified region and overlay type.

**Arguments:**

*   <region>: The geographic region (default: 'plains'). Valid options are: 'plains', 'ne', 'se', 'sw', 'nw'.
*   <overlay>: The type of overlay on the radar image (default: 'base'). Valid options are: 'base', 'totals', no input defaults to 'base'.
"""
	    
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
        response = requests.get(base_url, verify=False)  # Disable SSL verification
        response.raise_for_status()  # Still raise exceptions for other HTTP errors


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
        response = requests.get(storm_url, verify=False)  # Disable SSL verification
        response.raise_for_status()

        # Parse the HTML to extract image URLs
        soup = BeautifulSoup(response.content, 'html.parser')
        image_urls = extract_image_urls(soup)

        # Download and send images
        for image_url in image_urls:
            image_filename = image_url.split('/')[-1]
            urllib3.request.urlretrieve(image_url, image_filename)
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

ascat.help = """
**$ascat [storm_id]**

Fetches ASCAT (Advanced Scatterometer) images for the specified storm from the Fleet Numerical Meteorology and Oceanography Center (FNMOC).

**Arguments:**

*   `storm_id` (optional): The ID of the storm (e.g., '05L'). If not provided, the bot will list currently active storms.
"""

# --- Alerts Command ---
# state abbreviations turnt into Federal Information Processing Standards
state_abbreviations_to_fips = {
    'al': '01', 'ak': '02', 'az': '04', 'ar': '05', 'ca': '06', 'co': '08', 'ct': '09', 'de': '10', 
    'dc': '11', 'fl': '12', 'ga': '13', 'hi': '15', 'id': '16', 'il': '17', 'in': '18', 'ia': '19', 
    'ks': '20', 'ky': '21', 'la': '22', 'me': '23', 'md': '24', 'ma': '25', 'mi': '26', 'mn': '27', 
    'ms': '28', 'mo': '29', 'mt': '30', 'ne': '31', 'nv': '32', 'nh': '33', 'nj': '34', 'nm': '35', 
    'ny': '36', 'nc': '37', 'nd': '38', 'oh': '39', 'ok': '40', 'or': '41', 'pa': '42', 'ri': '43', 
    'sc': '45', 'sd': '46', 'tn': '47', 'tx': '48', 'ut': '49', 'vt': '50', 'va': '51', 'wa': '53', 
    'wv': '54', 'wi': '55', 'wy': '56', 
    # where you can't vote
    'as': '60', 'gu': '66', 'mp': '69', 'pr': '72', 'vi': '78'
}

@bot.command()
async def alerts(ctx, location: str = None):
    """Fetches and displays current weather alerts for a specified location or the user's location."""

    try:
	    
    location = location.lower()  # Convert input to lowercase for easier comparison, that way caps lock or something won't matter

        if location in state_abbreviations_to_fips:
            state_fips = state_abbreviations_to_fips[location]
            alerts_url = f"https://api.weather.gov/alerts/active?area={state_fips}" 
        else:
            # ... (handle other location types if needed, or provide an error message)
            await ctx.send("Invalid location. Please provide a two-letter state abbreviation (e.g., 'ga' for Georgia).")
            return

        print(f"Fetching alerts from: {alerts_url}") # Debugging statement
        response = requests.get(alerts_url)

        if response.status_code == 200:
            alerts_data = response.json()
            print(f"Alerts data received: {alerts_data}")  # Debugging statement

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

    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching alerts: {e}")
    except json.JSONDecodeError as e:
        await ctx.send(f"Error parsing alert data: {e}")
    except KeyError as e:
        await ctx.send(f"Unexpected data format in alerts response. Missing key: {e}")

alerts.help = """
Fetches and displays current weather alerts for a specified location or the user's location.

**Arguments:**

*   `location` (optional): The location for which you want to retrieve alerts.  Provide a two-letter state abbreviation (e.g., 'MT' for Montana). If not provided, the bot will attempt to use the user's location based on their Discord profile.
"""
# this was tested without bot.command() function, it works.

# --- Models Command --- 
@bot.command()
async def models(ctx, model_type: str, icao: str, variable: str):
    """Fetches model output for a specified airport, variable, and model type (deterministic or ensemble)."""

    try:
        icao = icao.upper()
        airport_data = airports.get(icao)
        if not airport_data:
            await ctx.send(f"Could not find airport with ICAO code {icao}.")
            return

        latitude = airport_data['lat']
        longitude = airport_data['lon']

        # Map shorthand parameters to things nobody wants to type
        if variable == 'rh':
            variable = 'relative_humidity_2m' #this needs to be expanded. absolutely zero people want to type something like et0_fao_evapotranspiration for their parameter. 
	if variable == 'temp':
	    variable = 'temperature_2m'
	if variable == 'dp':
	    variable = 'dew_point_2m'
	if variable == 'feelslike'
	    variable = 'apparent_temperature'

        # Available models
        available_deterministic_models = ["gfs_hrrr", "gfs_graphcast025"]
        available_ensemble_models = ["gfs025", "ecmwf_ifs025"]

        # Check model type and validity
        if model_type.lower() == 'det':
            if model.lower() not in available_deterministic_models:
                raise ValueError(f"Invalid deterministic model. Valid options are: {', '.join(available_deterministic_models)}")
            url = "https://api.open-meteo.com/v1/forecast"  # Deterministic API URL
        elif model_type.lower() == 'ens':
            if model.lower() not in available_ensemble_models:
                raise ValueError(f"Invalid ensemble model. Valid options are: {', '.join(available_ensemble_models)}")
            url = "https://ensemble-api.open-meteo.com/v1/ensemble"  # Ensemble API URL
        else:
            raise ValueError("Invalid model type. Please specify 'det' for deterministic or 'ens' for ensemble.")

        # Common parameters 
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", 
                "pressure_msl", "surface_pressure", "cloud_cover", "visibility", 
                "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", 
                "wind_direction_10m", "wind_gusts_10m"
            ],
            "timezone": "America/New_York",
            "wind_speed_unit": "kn",
            "precipitation_unit": "inch"
        }

        # Model-specific parameters
        if model_type.lower() == 'det':
            params["models"] = model.lower()
            params["forecast_hours"] = 168
        else:  # Ensemble
            params["models"] = model.lower()

        # Fetch data
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process hourly data (adjust based on model type if needed)
        hourly = response.Hourly()
        hourly_variables = list(map(lambda i: hourly.Variables(i), range(0, hourly.VariablesLength())))

        # Filter based on variable and altitude (adjust if needed for other variables)
        hourly_data_filter = filter(lambda x: x.Variable().name == variable and x.Altitude() == 2, hourly_variables)

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        # Process all members (for ensemble models)
        if model_type.lower() == 'ens':
            for var in hourly_data_filter:
                member = var.EnsembleMember()
                hourly_data[f"{variable}_member{member}"] = var.ValuesAsNumpy()
        else:  # Deterministic model
            for var in hourly_data_filter:
                hourly_data[variable] = var.ValuesAsNumpy()

        hourly_dataframe = pd.DataFrame(data=hourly_data)

        # Create and send embedded message
        model_type_str = "Deterministic" if model_type.lower() == 'det' else "Ensemble"
        embed = discord.Embed(title=f"{model_type_str} {model.upper()} Model Output for {icao} ({variable})", color=discord.Color.blue())
        for column in hourly_dataframe.columns:
            if column != 'date':
                embed.add_field(name=column, value=hourly_dataframe[column].to_string(index=False), inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"Error fetching model data: {e}")

models.help = """
**$models <model> <icao> <parameter>**

Fetches ensemble model output for a specified airport and variable.

**Arguments:**

*   `model`: The ensemble model to use. Valid options are: 'gfs025', 'ecmwf_ifs025'
*   `icao`: The ICAO code of the airport (e.g., 'KJFK', 'KATL').
*   `parameter`: The weather variable you want to see the model output for.  Currently supported parameters include:
    *   `temp`:  2-meter temperature (in °C)
    *   `rh`: 2-meter relative humidity (in %)
    *   `dp`: 2-meter dewpoint temperature (in °C)
    *   `feelslike`: Apparent temperature (in °C)
    *   `precipitation`: Precipitation (in inches)
    *   `rain`: Rain (in inches)
    *   `snowfall`: Snowfall (in inches)
    *   `snow_depth`: Snow depth (in inches)
    *   `pressure_msl`: Mean sea level pressure (in hPa)
    *   `surface_pressure`: Surface pressure (in hPa)
    *   `cloud_cover`: Cloud cover (in %)
    *   `visibility`: Visibility (in meters)
    *   `et0_fao_evapotranspiration`: FAO Evapotranspiration (in mm)
    *   `vapour_pressure_deficit`: Vapour pressure deficit (in hPa)
    *   `wind_speed_10m`: Wind speed at 10 meters (in knots)
    *   `wind_direction_10m`: Wind direction at 10 meters (in degrees)
    *   `wind_gusts_10m`: Wind gusts at 10 meters (in knots)

**Example:**

*   `$models ecmwf kvpc rh` (to get ECMWF model output for relative humidity at KVPC)
*   `$models gfs025 kjfk temp` (to get GFS model output for 2-meter temperature at KJFK)
"""

# revamped, not finished and probably wont work but whatever was in there before sucked. also what ens commented out was some cache nonsense so that way it wouldn't create another API request if you wanted the same product in a certain amount of time, which i guess who cares? lol

# --- Lightning Command ---
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Earth's radius in miles
    R = 3956 

    # Convert latitude and longitude to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Distance in miles
    distance = R * c
    
    return distance

@bot.command()
async def lightning(ctx, icao: str, radius: int = 5):
    """Checks for lightning strikes within a specified radius of an ICAO airport."""

    try:
        # 1. Get airport coordinates 
        airport_coords = get_airport_coordinates(icao)
        if airport_coords is None:
            await ctx.send(f"Could not find airport with ICAO code {icao}.")
            return

        # 2. Construct the API URL (replace with your actual credentials)
        api_url = f'https://data.api.xweather.com/lightning/{airport_coords[0]},{airport_coords[1]}?format=json&filter=cg&limit=10&radius={radius}&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET'

        # 3. Fetch lightning data
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # 4. Check for success and process data
        if data['success']:
            lightning_data = data['data']

            # 5. Check for strikes within the radius 
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

    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching lightning data: {e}")
    except KeyError as e:
        await ctx.send(f"Error parsing lightning data: {e}")
    except ValueError as e:
        await ctx.send(f"API Error: {e}") 

 
lightning.help = """
**$lightning <icao> [radius]**

Checks for lightning strikes within a specified radius of an ICAO airport.

**Arguments:**

*   `icao`: The ICAO code of the airport (e.g., 'KPIT', 'KBIX').
*   `radius` (optional): The radius (in miles) within which to check for lightning strikes (default: 5 miles).
"""
