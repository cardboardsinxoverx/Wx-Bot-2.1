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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
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
from siphon.simplewebservice.wyoming import WyomingUpperAir
import metpy
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, SkewT, Hodograph
import xarray as xr
# import requests_cache
from openmeteo_py import Options,OWmanager
# from retry import retry
import openmeteo_requests
from openmeteo_py import Hourly, Options
import airportsdata
import aiohttp
import asyncio

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

OPENWEATHERMAP_API_KEY = 'efd4f5ec6d2b16958a946b2ceb0419a6'

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
'''@bot.command()
async def restart(ctx):
    """Restarts the bot."""
    try:
        await ctx.send("Restarting...")
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            os.close(handler.fd)
        time.sleep(0.5)
        python = sys.executable
        os.execl(python, python, *sys.argv)
        await ctx.send("Back online bitches")
    except Exception as e:
        await ctx.send(f"Error during restart: {e}")'''

### --- Restart Command w/ logging & try-except block
@bot.command()
async def restart(ctx):
    """Restarts the bot."""
    try:
        await ctx.send("Restarting...")

        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            try:
                os.close(handler.fd)
            except Exception as e:
                print(f"Error closing handler: {e}")  # Log the error

        time.sleep(2)  # this needs to be adjusted, kind of like a distributor when you're adjusting timing on a motor. more time gives it longer to clean up tasks & operations and can increase reliability of the bot actually restarting, but can get to a point where its too slow.

        python = sys.executable
        os.execl(python, python, *sys.argv)

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

#def get_metar(icao, hoursback=0, format='json'):
  #  try:

     #   metar_observations = [data['rawText'] for data in json_data if 'rawText' in data]

     #   if not metar_observations:
      #      raise ValueError("No METAR observations found within the specified time range.")

     #   return metar_observations  # Return a list of METAR observations

  #  except requests.exceptions.RequestException as e:
        # Handle network errors during fetching
   #     raise Exception(f"Error fetching METAR data for {icao}: {e}")
 #   except (KeyError, ValueError) as e:
        # Handle potential parsing errors
  #      raise Exception(f"Error parsing METAR data for {icao}: {e}")

@bot.command()
async def metar(ctx, airport_code: str):
    """Sends latest metar for given airport"""
    airport_code = airport_code.upper()
    raw_metar = get_metar(airport_code)
    
    embed = discord.Embed(title=f"METAR for {airport_code}", description=raw_metar)
    await ctx.send(embed=embed)
    logging.info(f"User {ctx.author} requested METAR for {airport_code}")

#@bot.command()
#async def metar(ctx, airport_code: str, hoursback: int = 0): 
   # """Sends latest metar or multiple past METARs for given airport"""
   # airport_code = airport_code.upper()
   # raw_metars = get_metar(airport_code, hoursback)  

    # Handle multiple METARs
   # if hoursback > 0:
     #   for raw_metar in raw_metars:
      #      await ctx.send(f"`\n{raw_metar}\n`")
   # else:
    ##   await ctx.send(f"`\n{raw_metars[0]}\n`")  # Only the latest METAR

   # logging.info(f"User {ctx.author} requested METAR for {airport_code}

# what i have commented out is yet another attempt to pull more than one ob, i excluded the embed part and placed the text within ```these things```. -ln

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

'''# --- SkewT Command --- 
@bot.command()
async def skewt(ctx, station_code: str):
    """Fetches sounding data and generates a Skew-T diagram with various indices."""

    try:
        station_code = station_code.upper()

	# Fetch and process the sounding data
        # ds = xr.Dataset.from_dataframe(WyomingUpperAir.request_data(datetime.datetime.now(), station_code.strip('K')))

        # Get today's date in YYYY-MM-DD format
        now = datetime.datetime.today()

        # # Construct the URL 
        # sounding_url = f"https://weather.uwyo.edu/cgi-bin/sounding?region=naconf&STNM={station_code}&DATE={today}&HOUR=latest&YEAR={datetime.datetime.now().year}&ENV=std"

        # # Fetch the sounding data
        # response = requests.get(sounding_url, verify=False)
        # response.raise_for_status()

        # # Parse the HTML to extract the sounding text
        # soup = BeautifulSoup(response.content, 'html.parser')
        # sounding_data = soup.find("pre").text.strip()
        sounding_data = WyomingUpperAir.request_data(now, station_code)
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
        positive_shear = mpcalc.bulk_shear(profile.pres, profile.u, profile.v, height=slice(0, 3000 * units.m))
        srh = mpcalc.storm_relative_helicity(profile.u, profile.v, height, profile.storm_motion)
        total_totals = mpcalc.total_totals_index(profile.tmpc, profile.dwpc, profile.u, profile.v)
        # Assuming you have a function to calculate tropopause level 
        tropopause_level = mpcalc.calculate_tropopause_level(profile)

	# Calculate tropopause level using the birner function
        tropopause_level = mpcalc.birner(profile.pres, profile.tmpc, height=True)

        # Calculate wet-bulb temperature
        wet_bulb = mpcalc.wet_bulb_temperature(profile.pres, profile.tmpc, profile.dwpc)

        # Create the Skew-T plot
        fig = plt.figure(figsize=(8, 8))
        skew = SkewT(fig)

        # Pls plot
        skew.plot(profile.pres, profile.tempc, 'r')  # Temperature in red
        skew.plot(profile.pres, profile.dwpc, 'g')  # Dewpoint in green
        skew.plot(profile.pres, wet_bulb, 'b', linestyle='--')  # Wet-bulb temperature in dashed blue
        skew.plot_barbs(profile.pres[::2], profile.u[::2], profile.v[::2])  # Wind barbs 

        # mush indices on skewT
        plt.title(f'{station_code} {now.strftime("%Y-%m-%dT%H")} {profile.time[0].hour:02d}Z', weight='bold', size=20)
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
        height = profile.height * units.meter

	# add hodograph
        ax_hod = inset_axes(skew.ax, '25%', '20%', loc='upper left')
        h = Hodograph(ax_hod, component_range=80)  # Change range in windspeeds
        h.add_grid(increment=10)
        try:
            h.plot_colormapped(profile.u, profile.v, height)
 	 
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
        await ctx.send(f"Error retrieving/parsing Skew-T data for {station_code}: {e}. This could be happening for several reasons, such as network connection issues, timeout errors, data not being in the correct format, or the bot is requesting data it wasn't programmed to understand.")'''

# --- SkewT Command ---
@bot.command()
async def skewt(ctx, station_code: str, sounding_time: str = "12Z"):
    """Fetches 12Z or 00Z sounding data and generates a Skew-T diagram."""  # Updated docstring

    try:
        station_code = station_code.upper()
        sounding_time = sounding_time.upper()

        # Get today's date
        today = datetime.datetime.today()

        # Set sounding time based on user input
        if sounding_time == "12Z":
            now = datetime.datetime(today.year, today.month, today.day, 12, 0, 0, tzinfo=pytz.UTC)
        elif sounding_time == "00Z":
            # Handle the case where 00Z might be from the previous day
            if today.hour < 12:  # If it's before 12 PM local time, 00Z is from yesterday
                today -= datetime.timedelta(days=1)
            now = datetime.datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=pytz.UTC)
        else:
            raise ValueError("Invalid sounding time. Please choose either '12Z' or '00Z'.")

        # Fetch sounding data
        sounding_data = WyomingUpperAir.request_data(now, station_code)

        # Handle case where sounding data is not found
        if sounding_data is None or sounding_data.empty:
            raise ValueError("Sounding data not found. This is likely because the sounding balloon was not released. Please check a neighboring WMO or try again later.")

        # Convert sounding data to MetPy units
        p = sounding_data['pressure'].values * units.hPa
        T = sounding_data['temperature'].values * units.degC
        Td = sounding_data['dewpoint'].values * units.degC
        u = sounding_data['u_wind'].values * units.knots
        v = sounding_data['v_wind'].values * units.knots
        hght = sounding_data['height'].values * units.meter

        # Calculate wet-bulb temperature
        wet_bulb = mpcalc.wet_bulb_temperature(p, T, Td)

        # Create the Skew-T plot using MetPy
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig)
        skew.plot(p, T, 'r')
        skew.plot(p, Td, 'g')
        skew.plot(p, wet_bulb, 'b', linestyle='--')  
        skew.plot_barbs(p, u, v)
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)

        # Add the parcel profile
        #skew.plot(parcel_profile, 'k', linewidth=2)  # Add the parcel path

        # Shade areas of CAPE and CIN
        #skew.shade_cape(p, T, parcel_profile)
        #skew.shade_cin(p, T, parcel_profile)

        # Add labels and title
        plt.title(f'{station_code} {now.strftime("%Y-%m-%d %HZ")}', weight='bold', size=14, color='#556B2F')

        # Add indices to the plot
        #skew.ax.text(0.7, 0.1, f'CAPE: {cape.to("J/kg"):.0f} J/kg', transform=skew.ax.transAxes)
       # skew.ax.text(0.7, 0.05, f'CIN: {cin.to("J/kg"):.0f} J/kg', transform=skew.ax.transAxes)
       # skew.ax.text(0.7, 0.15, f'LIFTED INDEX: {lifted_index:.0f}', transform=skew.ax.transAxes)
       # skew.ax.text(0.7, 0.2, f'PWAT: {pwat.to("inch"):.2f} in', transform=skew.ax.transAxes)

        # Save and send the Skew-T diagram
        temp_image_path = f"skewt_{station_code}_observed.png"
        plt.savefig(temp_image_path, format='png')
        plt.close(fig)
        await ctx.send(file=discord.File(temp_image_path))
        os.remove(temp_image_path)

    # Enhanced error handling
    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching sounding data for {station_code}: {e}. Please check your network connection or try again later.")
    except AttributeError as e:
        await ctx.send(f"Error processing sounding data for {station_code}: {e}. The data might be incomplete or in an unexpected format.")
    except ValueError as e:
        await ctx.send(e)
    except Exception as e:
        await ctx.send(f"An unexpected error occurred while generating the Skew-T for {station_code}: {e}")
    #except Exception as e:
        #await ctx.send(f"Error calculating indices: {e}")
       # return  # Stop execution if there's an error in calculations
	    
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
            "fulldisk": {1: "GeoColor (True Color)", 2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 5: "RGB Air Mass"},  # Changed "airmass" to 5
            "mesosector": {1: "GeoColor (True Color)", 2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
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
                    5: "https://whirlwind.aos.wisc.edu/~wxp/goes16/multi_air_mass_rgb/fulldisk/latest_fulldisk_1.jpg"
                },
                "mesosector": {
                    13: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_ircm/latest_meso_1.jpg",
                    9: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_wvc/latest_meso_1.jpg",
                    2: "https://whirlwind.aos.wisc.edu/~wxp/goes16/grb/meso_vis/latest_meso_1.jpg"
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

        valid_regions = ["chase", "ne", "se", "sw", "nw", "pr"]
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
	    ("pr", "base"): "https://tempest.aos.wisc.edu/radar/pr3comp.gif",  # Added PR base link
            ("pr", "totals"): "https://tempest.aos.wisc.edu/radar/prPcomp.gif"  # Added PR totals link
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
	    
def add_map_overlay(ax, lat=None, lon=None, icon_path=None, logo_path="https://github.com/cardboardsinxoverx/Wx-Bot-2.1/blob/main/photo.jpg", zoom=0.1):
    """Adds a marker (if lat/lon provided) and the Marine Weather emblem (from a URL) to the map image."""

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

    # 2. Add Marine Weather Emblem (from URL)
    try:
        response = requests.get(logo_path)
        response.raise_for_status() 
        logo_img = Image.open(BytesIO(response.content))
    except (requests.exceptions.RequestException, PIL.UnidentifiedImageError) as e:
        logging.error(f"Error loading Marine Weather emblem from URL: {e}")
        return

    logo_img.thumbnail((25, 25)) 

    # Calculate logo position (5x5 pixels from bottom right)
    fig = ax.get_figure()
    fig_width, fig_height = fig.get_size_inches() * fig.dpi

    x_pos = fig_width - logo_img.width - 5  
    y_pos = 5

    logo = OffsetImage(logo_img)
    ab_logo = AnnotationBbox(logo, (x_pos, y_pos),
                             frameon=False,
                             xycoords='figure pixels',  # Position in figure pixels
                             boxcoords="offset points",
                             box_alignment=(1, 0),
                             zorder=10)
    ax.add_artist(ab_logo)
# ens before this works, the file names and source maybe for it? needs to be correct. this was just some generic code written with "logo.png". I dont know what the bots avatar's name is and if this just isn't possible or its just too silly, then I'll revert the code. I just thought you'd be able to make it work from here 

# --- ASCAT Command ---
'''@bot.command()
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
    return [base_url + img['src'] for img in image_tags]'''
# working ASCAT command for Atlantic Basin

# --- ASCAT Command ---
@bot.command()
async def ascat(ctx, storm_id: str = None):
    """Fetches ASCAT images for the specified storm from FNMOC across all basins."""

    try:
        # Fetch the main FNMOC TCWEB page 
        base_url = "https://www.fnmoc.navy.mil/tcweb/cgi-bin/tc_home.cgi"  
        response = requests.get(base_url, verify=False) 
        response.raise_for_status()

        # Parse the HTML to find active storms across all basins
        active_storms = extract_active_storms_global(response.content)

        if storm_id is None:
            # If no storm_id is provided, list the active storms
            if active_storms:
                storm_list = [f"{s['id']} ({s['basin']})" for s in active_storms]
                await ctx.send(f"Currently active storms: {', '.join(storm_list)}")
            else:
                await ctx.send("No active storms found.")
            return  # Exit the command early

        # Check if the requested storm is active (across all basins)
        matching_storms = [s for s in active_storms if s['id'].upper() == storm_id.upper()]
        if not matching_storms:
            raise ValueError(f"Storm '{storm_id}' not found among active storms.")

        # Get basin information for the specified storm
        storm_basin = matching_storms[0]['basin']

        # Construct the URL for the storm's ASCAT image page (adjust based on basin)
        storm_url = f"{base_url}?YEAR=2024&MO=Aug&BASIN={storm_basin}&STORM_NAME={storm_id}&SENSOR=&PHOT=yes&ARCHIVE=Mosaic&NAV=tc&DISPLAY=all&MOSAIC_SCALE=20%&STYLE=table&ACTIVES={','.join([s['id'] for s in active_storms])}&TYPE=ascat&CURRENT=LATEST.jpg&PROD=hires&DIR=/tcweb/dynamic/products/tc24/{storm_basin}/{storm_id}/ascat/hires&file_cnt=160"

        # Fetch the storm's ASCAT image page 
        response = requests.get(storm_url, verify=False) 
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

def extract_active_storms_global(html_content):
    """Parses the HTML content to extract a list of active storms across all basins."""
    soup = BeautifulSoup(html_content, 'html.parser')
    active_storms_table = soup.find('table', {'id': 'StormList'}) 

    if active_storms_table:
        storms = []
        for row in active_storms_table.find_all('tr')[1:]:  # Skip the header row
            columns = row.find_all('td')
            if len(columns) >= 2: 
                storm_id = columns[0].text.strip()
                basin = columns[1].text.strip() 
                storms.append({'id': storm_id, 'basin': basin})
        return storms
    else:
        return []


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
# future ASCAT command for global storms

# --- Alerts Command --- 
@bot.command()
async def alerts(ctx, state_abbr: str = None):
    """Fetches and displays current weather alerts for a specified state or the user's location."""

    try:
        state_abbr = state_abbr.upper() if state_abbr else None  # Handle potential None value and ensure uppercase

        if state_abbr:
            alerts_url = f"https://api.weather.gov/alerts/active?area={state_abbr}"
        else:
            state_abbr = await get_user_location(ctx) 
            if state_abbr:
                alerts_url = f"https://api.weather.gov/alerts/active?area={state_abbr}"
            else:
                await ctx.send("Unable to determine your location. Please provide a two-letter state abbreviation.")
                return

        print(f"Fetching alerts from: {alerts_url}") 
        response = requests.get(alerts_url)
        response.raise_for_status()  # Raise an exception for bad HTTP status codes

        alerts_data = response.json()
        print(f"Alerts data received: {alerts_data}") 

        filtered_alerts = [
            alert for alert in alerts_data.get('features', []) 
            if alert.get('properties') and alert['properties'].get('event') and alert['properties'].get('severity')
        ]

        if filtered_alerts:
            for alert in filtered_alerts:
                properties = alert['properties']
                embed = discord.Embed(title=properties['headline'], color=discord.Color.red())
                embed.add_field(name="Severity", value=properties['severity'], inline=True)
                embed.add_field(name="Effective", value=properties['onset'], inline=True)
                embed.add_field(name="Expires", value=properties['expires'], inline=True)

                # Clean up and format the area descriptions
                cleaned_area_desc = [area.strip() for area in properties['areaDesc'] if area.strip()]
                area_desc = "; ".join(cleaned_area_desc) 
                embed.add_field(name="Area", value=area_desc, inline=False)

                embed.add_field(name="Description", value=properties['description'], inline=False)
                embed.add_field(name="Instructions", value=properties['instruction'] or "None", inline=False)
                await ctx.send(embed=embed)
        else:
            await ctx.send("No weather alerts found for the specified state.")

    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching alerts: {e}")
    except json.JSONDecodeError as e:
        await ctx.send(f"Error parsing alert data: {e}")
    except KeyError as e:
        await ctx.send(f"Unexpected data format in alerts response. Missing key: {e}")

alerts.help = """
Fetches and displays current weather alerts for a specified location or the user's location.

**Arguments:**

*   `location` (optional): The location for which you want to retrieve alerts. Provide a two-letter state abbreviation (e.g., 'MT' for Montana). If not provided, the bot will attempt to use the user's location based on their Discord profile.
"""
# correctted URL 

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

       # Map shorthand parameters 
        shorthand_map = {
            'rh': 'relative_humidity_2m',
            'temp': 'temperature_2m',
            'dp': 'dew_point_2m',
            'feelslike': 'apparent_temperature',
            'precip': 'precipitation',
            'rain': 'rain',
            'snow': 'snowfall',
            'snowdepth': 'snow_depth',
            'mslp': 'pressure_msl',
            'sp': 'surface_pressure',
            'clouds': 'cloud_cover',
            'vis': 'visibility',
            'et': 'et0_fao_evapotranspiration',
            'vpd': 'vapor_pressure_deficit',
            'ws': 'wind_speed_10m',
            'wd': 'wind_direction_10m',
            'gusts': 'wind_gusts_10m'
        }

        if variable.lower() in shorthand_map:
            variable = shorthand_map[variable.lower()]
	

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
                "et0_fao_evapotranspiration", "vapor_pressure_deficit", "wind_speed_10m", 
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
    *   `precip`: Precipitation (in inches)
    *   `rain`: Rain (in inches)
    *   `snow`: Snowfall (in inches)
    *   `snowdepth`: Snow depth (in inches)
    *   `mslp`: Mean sea level pressure (in hPa)
    *   `sp`: Surface pressure (in hPa)
    *   `clouds`: Cloud cover (in %)
    *   `vis`: Visibility (in meters)
    *   `et`: FAO Evapotranspiration (in mm)
    *   `vpd`: Vapor pressure deficit (in hPa)
    *   `ws`: Wind speed at 10 meters (in knots)
    *   `wd`: Wind direction at 10 meters (in degrees)
    *   `gusts`: Wind gusts at 10 meters (in knots)

**Example:**

*   `$models ecmwf kvpc rh` (to get ECMWF model output for relative humidity at KVPC)
*   `$models gfs025 kjfk temp` (to get GFS model output for 2-meter temperature at KJFK)
"""

# revamped, not finished and probably wont work but whatever was in there before sucked. also what ens commented out was some cache nonsense so that way it wouldn't create another API request if you wanted the same product in a certain amount of time, which i guess who cares? lol

# --- Lightning Command --- 
def get_airport_coordinates(icao):
  """
  Fetches latitude and longitude coordinates for a given ICAO airport code using the OpenFlights API.
  
  Args:
      icao (str): The ICAO code of the airport.

  Returns:
      tuple: A tuple containing the latitude and longitude coordinates (float, float) if found, otherwise None.
  """

  api_url = f'https://openflights.org/api/v1/airports/{icao}'
  try:
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()

    if data and 'latitude' in data and 'longitude' in data:
      return (float(data['latitude']), float(data['longitude']))
    else:
      return None

  except requests.exceptions.RequestException as e:
    print(f"Error fetching airport data: {e}")
    return None

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

        # 2. Construct the API URL 
        api_url = f'https://data.api.xweather.com/lightning/{airport_coords[0]},{airport_coords[1]}?format=json&filter=cg&limit=10&radius={radius}&client_id=LI9ra7oPstcUiVqCpw2NB&client_secret=vLc8FjGxYpkiXMunhvUwoVVKlBOizDmuzseYX0dB'

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

# @lightning.help
# async def lightning_help(ctx):
#     await ctx.send("""
#         Checks for lightning strikes near an airport.

#         **Usage: $lighting <icao> <distance(sm)>**
#         """)

# --- Meteogram Command ---
@bot.command()
async def meteogram(ctx, icao: str, hoursback: str = None):
    """
    Generates a meteogram for the given ICAO code.
    """

    # Create the figure outside the try block
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 18), dpi=150, sharex=True)
    fig.patch.set_facecolor('white')

    try:
        # Call the meteogram function in an asyncio task to avoid blocking
        loop = asyncio.get_event_loop()
        fname = await loop.run_in_executor(None, meteogram, icao, hoursback)

        if fname is None or not os.path.exists(fname):
            await ctx.send(f"Failed to generate meteogram for {icao}.")
        else:
            await ctx.send(f'Meteogram for {icao} created.')
            await ctx.send(file=discord.File(fname))

    except ValueError as e:
        await ctx.send(str(e))  # Send the specific error message
    except Exception as e:
        await ctx.send(f'Error generating meteogram for {icao}: {e}')

def get_metar_meteogram(icao, hoursback=None):

    if hoursback:
        metar_url = f'https://www.aviationweather.gov/metar/data?ids={icao}&format=raw&date=&hours={hoursback}&taf=off'
    else:
        metar_url = f'https://www.aviationweather.gov/metar/data?ids={icao}&format=raw&date=&hours=0&taf=off'
    src = requests.get(metar_url).content
    soup = BeautifulSoup(src, "html.parser")
    metar_data = soup.find(id='awc_main_content_wrap')

    obs = ''
    for i in metar_data:
        if str(i).startswith('<code>'):
            line = str(i).lstrip('<code>').rstrip('</code>')
            obs+=line
            obs+='\n'
    return obs

def meteogram(icao, hoursback):
    icaos = [icao.upper()]

    for i, icao in enumerate(icaos):
        txt = get_metar_meteogram(icao, hoursback).split('\n')[:-1]
        if not txt:  # Check if txt is empty
            raise ValueError(f"No METAR data found for {icao}.")

        if i == 0:
            df = parse_metar_to_dataframe(txt[-1])
        for row in txt[::-1]:
            df = df.append(parse_metar_to_dataframe(row))

    df['tempF'] =  (df['air_temperature'] * 9/5) + 32
    df['dewF'] =  (df['dew_point_temperature'] * 9/5) + 32
    df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)

    ## nan HI values where air temp < 80 F
    df.loc[df['tempF'] < 80, ['heat_index']] = np.nan

    try:
        df['wet_bulb'] = mpcalc.wet_bulb_temperature(
            df['air_pressure_at_sea_level'].values * units('hPa'),
            df['air_temperature'].values * units('degC'),
            df['dew_point_temperature'].values * units('degC')
        ).to('degF').m
    except ValueError as e:
        print("Can't calculate wet bulb")
        df['wet_bulb'] = np.nan

    WNDDIR = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
    WNDDEG = np.arange(0, 361,
 22.5)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 18), dpi=150, sharex=True)
    fig.patch.set_facecolor('white')

    # Plot on ax1 (Temperature)
    ax1.plot(df['date_time'], df['tempF'], label='<span class="math-inline">T</span>', linestyle='-', marker='', color='tab:red')
    ax1.plot(df['date_time'], df['wet_bulb'], label='<span class="math-inline">T\_w</span>', linestyle='-', marker='', color='tab:blue')
    ax1.plot(df['date_time'], df['dewF'], label='<span class="math-inline">T\_d</span>', linestyle='-', marker='', color='tab:green')
    ax1.plot(df['date_time'], df['heat_index'], label='<span class="math-inline">RF</span>', linestyle='-', marker='', color='tab:orange')
    # Removed wind chill plot

    ax1.set_ylabel('Temperature (°F)')
    ax1.set_xlabel('Z-Time (MM-DD HH)')
    ax1.set_title(f"{df['station_id'][0]}\n{df['date_time'][0].strftime('%Y-%m')}")
    ax1.set_xticks(pd.date_range(df['date_time'][0].strftime('%Y-%m-%d %H'), df['date_time'][-1].strftime('%Y-%m-%d %H'), freq='2H'))
    ax1.set_xticklabels(pd.date_range(df['date_time'][0].strftime('%Y-%m-%d %H'), df['date_time'][-1].strftime('%Y-%m-%d %H'), freq='2H').strftime('%m-%d %H%MZ'))
    ax1.grid(which='both')

    ax1.grid(which='major', axis='x', color='black')
    ax1.legend(loc='upper left')

    ax2b = ax2.twinx()
    ax2.plot(df['date_time'], df['wind_speed']*1.15078, label='Speed', linestyle='-', marker='', color='tab:blue')
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    ax2b.plot(df['date_time'], df['wind_direction'], label='Direction', linestyle='', marker='*', color='tab:cyan')
    lines_2, labels_2 = ax2b.get_legend_handles_labels()
    max_wind = df['wind_speed'].max()*1.15078
    if max_wind > 30:
        ax2.set_ylim([0, max_wind+5])
    else:
        ax2.set_ylim([0, 30])
    ax2.set_ylabel('Wind Speed (mph)')
    ax2b.set_ylabel('Wind Direction (°)')
    ax2b.set_ylim([-10,370])
    ax2b.set_yticks(WNDDEG[::4])
    ax2b.set_yticklabels(WNDDIR[::4])
    ax2.grid(which='both')
    ax2.grid(which='major', axis='x', color='black')
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax2.legend(lines, labels, loc='upper left')


    ax3.plot(df['date_time'], df['altimeter'], label='Altimeter', linestyle='-', marker='', color='tab:brown')
    ax3.set_ylabel('Pressure (inHg)')
    # ax3.set_ylim([29.70, 30.10])
    ax3.grid(which='both')
    ax3.grid(which='major', axis='x', color='black')


    ax4.plot(df['date_time'], df['low_cloud_level']/1000, label='Low', linestyle='', marker='*')
    ax4.plot(df['date_time'], df['medium_cloud_level']/1000, label='Medium', linestyle='', marker='*')
    ax4.plot(df['date_time'], df['high_cloud_level']/1000, label='High', linestyle='', marker='*')
    ax4.plot(df['date_time'], df['highest_cloud_level']/1000, label='Highest', linestyle='', marker='*')
    ax4.set_ylim([0, 30])
    ax4.set_ylabel('Cloud Height (kft)')
    ax4.set_xlabel('Date (MM-DD-HH Z)')
    ax4.set_xticks(pd.date_range(df['date_time'][0].strftime('%Y-%m-%d %H'), (df['date_time'][-1]+timedelta(hours=6)).strftime('%Y-%m-%d %H'), freq='6H'))
    ax4.set_xticklabels(pd.date_range(df['date_time'][0].strftime('%Y-%m-%d %H'), (df['date_time'][-1]+timedelta(hours=6)).strftime('%Y-%m-%d %H'), freq='6H').strftime('%d %HZ'))
    ax4.legend(loc='upper left')
    ax4.xaxis.set_major_locator(DayLocator())
    ax4.xaxis.set_minor_locator(HourLocator(range(0, 25, 3)))
    ax4.grid(which='both')
    ax4.grid(which='major', axis='x', color='black')
#   ax4.grid(which='minor', axis='x', linewidth=0.5)
    ax4.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_minor_formatter(DateFormatter('%H'))
    fig.autofmt_xdate(rotation=50)
    fname = f"../imgs/meteogram/metorgram_{df['station_id'][0]}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    return fname

# --- Meteogram Generation ---
def meteogram(icao, hoursback):
    # Calculate wind chill and heat index (assuming you have the `to_wind_chill` and `to_heat_index` functions defined)
    #df['wind_chill'] = to_wind_chill(df['tempF'].values, df['wind_speed'].values)
    #df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)

    # Handle missing values (replace NaNs with a specific value or drop rows)
    #df.fillna(value={'wind_chill': -999, 'heat_index': -999}, inplace=True)  # Replace NaNs with -999
    # or
    #df.dropna(subset=['wind_chill', 'heat_index'], inplace=True)  # Drop rows with NaN values

    try:
        # 3. Generate the meteogram plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 18), dpi=150, sharex=True)
        fig.patch.set_facecolor('white')

        # Calculate wind chill and heat index
        #df['wind_chill'] = to_wind_chill(df['tempF'].values, df['wind_speed'].values)
        #df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)

        # Plot temperature, dew point, etc. on ax1
        ax1.plot(df['date_time'], df['tempF'], label='T', color='red')
        ax1.plot(df['date_time'], df['dewF'], label='Td', color='green')
        # ... other plots on ax1 ...

        ax1.set_ylabel('Temperature (°F)')
        ax1.legend()

        # Plot wind speed and direction on ax2
        ax2.plot(df['date_time'], df['wind_speed'], label='Wind Speed', color='blue')
        # ... (potentially add wind barbs or other wind visualizations)

        ax2.set_ylabel('Wind Speed')
        ax2.legend()

        # Plot pressure on ax3
        ax3.plot(df['date_time'], df['altimeter'], label='Altimeter', color='brown')

        ax3.set_ylabel('Pressure (inHg)')
        ax3.legend()

        # Plot cloud cover or other variables on ax4
        ax4.plot(df['date_time'], df['cloud_cover'], label='Cloud Cover (%)', color='gray')
        ax4.plot(df['date_time'], df['cloud_cover'], label='Cloud Cover (%)', color='gray')

        ax4.set_ylabel('Cloud Cover (%)')  # Or adjust the label based on your plot
        ax4.legend()

        # Format x-axis (shared by all subplots)
        ax4.set_xlabel('Time')

        # 4. Save the plot as an image
        # Adjust the file path as needed
        temp_image_path = f"meteogram_{icao}.png"  # Or use an absolute path
        plt.savefig(temp_image_path, bbox_inches='tight')
        plt.close(fig)

        return temp_image_path

    except Exception as e:
        print(f"Error generating meteogram image for {icao}: {e}")
        return None  # Return None to indicate an error

# --- Time Command ---
@bot.command(name='utc')
async def worldtimes(ctx):

  embed = discord.Embed(title="**Time Zones**", color=0x007F00)

  utc_now = pytz.utc.localize(datetime.datetime.utcnow())

  # define time zones
  us_timezones = {
      "Hawaii": "Pacific/Honolulu",
      "Alaska": "America/Anchorage",
      "Pacific": "America/Los_Angeles",
      "Mountain": "America/Denver",
      "Central": "America/Chicago",
      "Eastern": "America/New_York"
  }

  international_timezones = {
      "London": "Europe/London",
      "Berlin": "Europe/Berlin",
      "Tokyo": "Asia/Tokyo",
      "Sydney": "Australia/Sydney",
      "Tehran (Iran)": "Asia/Tehran",
      "Jerusalem (Israel)": "Asia/Jerusalem",
      "Moscow": "Europe/Moscow",
      "Beijing": "Asia/Shanghai"
  }

  #add the times to the embed
  for region, timezone_str in us_timezones.items():
      timezone = pytz.timezone(timezone_str)
      local_time = utc_now.astimezone(timezone)
      embed.add_field(name=f"{region} (US)", value=local_time.strftime('%H:%M:%S'), inline=True)

  for city, timezone_str in international_timezones.items():  # Use 'international_timezones' here
      timezone = pytz.timezone(timezone_str)
      local_time = utc_now.astimezone(timezone)
      embed.add_field(name=city, value=local_time.strftime('%H:%M:%S'), inline=True)

  await ctx.send(embed=embed)

# --- Fcst Command --- 
@bot.command()
async def forecast(ctx, *, location: str):
    """Provides a weather forecast for the next few days for a given location."""
    url = f'https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHERMAP_API_KEY}&units=imperial'  # Assuming you want Fahrenheit
    response = requests.get(url)
    data = response.json()

    if data['cod'] == '200':
        embed = discord.Embed(title=f"Weather Forecast for {data['city']['name']}", color=0x007F00)
        for forecast in data['list'][:4]:  # Get forecast for the next few days (adjust as needed)
            timestamp = forecast['dt']
            date = datetime.datetime.fromtimestamp(timestamp).strftime('%A, %B %d')
            temperature = forecast['main']['temp']
            condition = forecast['weather'][0]['description']
            embed.add_field(name=date, value=f"Temperature: {temperature}°F\nCondition: {condition}", inline=False)
        await ctx.send(embed=embed)
    else:
        await ctx.send(f"dad dammit")

# --- Air Sucks Command --- 
@bot.command()
async def airquality(ctx, *, location: str):
    """Shows the air quality index (AQI) and related information for a specific location."""
    url = f'https://api.openweathermap.org/data/2.5/air_pollution?q={location}&appid={OPENWEATHERMAP_API_KEY}'
    response = requests.get(url)
    data = response.json()

    if data['cod'] == '200':
        aqi = data['list'][0]['main']['aqi']
        components = data['list'][0]['components']
        embed = discord.Embed(title=f"Air Quality in {location}", color=0x007F00)
        embed.add_field(name="AQI", value=aqi, inline=True)
        embed.add_field(name="Main Pollutants", value=", ".join(f"{k}: {v}" for k, v in components.items()), inline=False)
        await ctx.send(embed=embed)
    else:
        await ctx.send(f"dad dammit")
	    
if __name__ == "__main__":
    bot.run(token=config.DISCORD_TOKEN)
