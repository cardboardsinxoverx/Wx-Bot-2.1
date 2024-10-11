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
import pytz
import shapefile
import zipfile
from bs4 import BeautifulSoup  # Instead of 'import BeautifulSoup'
from io import BytesIO
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import os
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
#from siphon.simplewebservice.adds import ADDS
import metpy
from metpy.units import units
import matplotlib.gridspec as gridspec
from metpy.plots import add_metpy_logo, SkewT, Hodograph
import xarray as xr
from openmeteo_py import Options,OWmanager
import openmeteo_requests
from openmeteo_py import Hourly, Options
import aiohttp
import asyncio
from metpy.calc import parcel_profile, mixed_layer_cape_cin
import certifi
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import ColdFront, WarmFront, OccludedFront, StationaryFront, add_metpy_logo, StationPlot
from shapely.geometry import Point, LineString
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
import metpy.calc as mpcalc
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from metpy.plots import add_metpy_logo
from metpy.calc import dewpoint_from_relative_humidity
from metpy.cbook import get_test_data
from utils import parse_date
from weather_calculations import calc_mslp
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time
import re
import xmltodict
from fuzzywuzzy import fuzz
import io
import meteogram as mg
import metpy.interpolate as mpinterpolate
from matplotlib.colors import LinearSegmentedColormap
from skewt import skewt  # Import the skewt command function
from siphon.simplewebservice.iastate import IAStateUpperAir
from datetime import datetime, timedelta, timezone
from metpy.plots import add_timestamp
from scipy.signal import find_peaks
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from pyfonts import load_font
import matplotlib.dates as mdates
from tropycal import realtime, tracks, rain, recon
from uuid import uuid4
from timezonefinder import TimezoneFinder

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)


# --- on_message Event Handler ---
''''@bot.event
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
    await bot.process_commands(message)  # Process bot commands'''
	    
# --- git pull command ---
@bot.command()
async def pull(ctx):
    await ctx.send("Pulling down changes from GitHub...")
    res = subprocess.run(['git', 'pull'], capture_output=True, text=True)
    await ctx.send(f'{res}\nFinished. Good luck.')

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

        # Extract raw METAR observations
        raw_metars = [entry['rawOb'] for entry in json_data]  # Revert back to 'rawOb'

        return raw_metars

    except requests.exceptions.RequestException as e:
        # Handle network errors during fetching
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        # Handle potential parsing errors
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

@bot.command()
async def metar(ctx, airport_code: str, hoursback: int = 0):
    """Sends latest metar for given airport, optionally specifying hours back, formatted within code blocks."""
    airport_code = airport_code.upper()
    raw_metars = get_metar(airport_code, hoursback)

    # Handle multiple observations if hoursback is specified
    if hoursback > 0:
        embed = discord.Embed(title=f"METARs for {airport_code} (Last {hoursback} Hours)", color=0x01c0db)
        for i, raw_metar in enumerate(raw_metars):
            embed.add_field(name=f"Observation {i+1}", value=f"`\n{raw_metar}\n`", inline=False)  # Wrap in code block
    else:
        embed = discord.Embed(title=f"METAR for {airport_code}", description=f"`\n{raw_metars[0]}\n`", color=0x01c0db)  # Wrap in code block

    await ctx.send(embed=embed)
    logging.info(f"User {ctx.author} requested METAR for {airport_code} (hoursback={hoursback})")

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

# --- Meteogram Command ---
def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)

        # Check if any METAR data was found at all
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        # Extract raw METAR observations
        raw_metars = [entry['rawOb'] for entry in json_data]
        return raw_metars

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

# Function to handle cloud cover and vertical visibility (VVxxx)
def extract_cloud_info(metar):
    cloud_levels = {
        "low": [],
        "mid": [],
        "high": [],
        "vertical_visibility": None
    }

    # Regular expression for matching cloud cover and vertical visibility
    cloud_pattern = re.compile(r'((FEW|SCT|BKN|OVC)(\d{3}))|(VV(\d{3}))')

    # Search for cloud layers or vertical visibility in the METAR
    cloud_matches = re.findall(cloud_pattern, metar)

    for match in cloud_matches:
        if match[1]:  # Cloud layer (FEW, SCT, BKN, OVC)
            cover = match[1]
            altitude_hundreds = int(match[2])  # in hundreds of feet
            altitude_ft = altitude_hundreds * 100  # convert to feet

            # Categorize cloud levels based on altitude
            if altitude_ft <= 6500:
                cloud_levels["low"].append((cover, altitude_ft))
            elif 6500 < altitude_ft <= 20000:
                cloud_levels["mid"].append((cover, altitude_ft))
            else:
                cloud_levels["high"].append((cover, altitude_ft))

        if match[4]:  # Vertical visibility (VVxxx)
            vv_hundreds = int(match[4])
            cloud_levels["vertical_visibility"] = vv_hundreds * 100  # convert to feet

    return cloud_levels

# Function to convert pressure from altimeter setting (Axxxx) to inches of mercury
def convert_pressure(altimeter_str):
    return float(altimeter_str[1:]) / 100  # 'Axxxx' is in hundredths of inHg

# Function to extract wind information from a METAR string
def extract_wind_info(metar):
    wind_direction = -999
    wind_speed = -999
    wind_gusts = np.nan  # Initialize as NaN in case there's no gust

    wind_match = re.search(r'(\d{3})(\d{2})(G\d{2})?KT', metar)
    if wind_match:
        wind_direction = int(wind_match.group(1))
        wind_speed = int(wind_match.group(2))
        if wind_match.group(3):
            wind_gusts = int(wind_match.group(3)[1:])  # Extract gust speed without 'G'

    # Ensure no negative values for wind speeds or gusts
    wind_speed = max(wind_speed, 0)
    wind_gusts = max(wind_gusts, 0) if not np.isnan(wind_gusts) else np.nan

    return wind_direction, wind_speed, wind_gusts

def process_metar_data(metar_list):
    data = {
        "time": [],
        "temperature": [],
        "dewpoint": [],
        "wind_direction": [],
        "wind_speed": [],
        "wind_gusts": [],
        "pressure": [],
        "low_clouds": [],
        "mid_clouds": [],
        "high_clouds": [],
        "vertical_visibility": []  # To store VVxxx data
    }

    for metar in metar_list:
        parts = metar.split()

        # Extract observation time
        try:
            observation_time_str = parts[1]
            observation_time = datetime.strptime(observation_time_str, '%d%H%MZ').replace(tzinfo=timezone.utc)
            data["time"].append(observation_time)
        except (IndexError, ValueError):
            continue

        # Temperature and dewpoint extraction
        temp = None
        dewpoint = None
        temp_dewpoint_match = re.search(r'([M]?\d{1,2})/([M]?\d{1,2})', metar)
        if temp_dewpoint_match:
            temp_str, dewpoint_str = temp_dewpoint_match.groups()
            temp = int(temp_str) if temp_str and temp_str[0] != 'M' else None
            dewpoint = int(dewpoint_str) if dewpoint_str and dewpoint_str[0] != 'M' else None

            # Ensure temperature is not lower than dewpoint
            if temp is not None and dewpoint is not None:
                if temp < dewpoint:
                    temp = dewpoint

        # Only append valid (non-None) values
        if temp is not None and dewpoint is not None:
            data["temperature"].append(temp)
            data["dewpoint"].append(dewpoint)
        else:
            # Handle missing values by appending NaN
            data["temperature"].append(np.nan)
            data["dewpoint"].append(np.nan)

        # Wind information
        direction, speed, gusts = extract_wind_info(metar)
        data["wind_direction"].append(direction)
        data["wind_speed"].append(speed)
        data["wind_gusts"].append(gusts)

        # Pressure extraction
        pressure = -999
        pressure_match = re.search(r'A(\d{4})', metar)
        if pressure_match:
            pressure = convert_pressure(pressure_match.group(0))
        data["pressure"].append(pressure)

        # Cloud cover and vertical visibility extraction
        low_cloud = mid_cloud = high_cloud = vertical_visibility = np.nan
        cloud_patterns = {
            'low': r'(FEW|SCT|BKN|OVC)(\d{3})',
            'vv': r'VV(\d{3})'  # Vertical visibility pattern
        }

        # Process clouds and VV
        for part in parts:
            low_cloud_match = re.match(cloud_patterns['low'], part)
            vv_match = re.match(cloud_patterns['vv'], part)

            if low_cloud_match:
                cover_type, height = low_cloud_match.groups()
                height = int(height) * 100  # Convert to feet

                if cover_type == 'FEW' and np.isnan(low_cloud):
                    low_cloud = height
                elif cover_type == 'SCT':
                    mid_cloud = height
                elif cover_type == 'BKN' or cover_type == 'OVC':
                    high_cloud = height

            if vv_match:
                vertical_visibility = int(vv_match.group(1)) * 100  # Convert to feet

        data["low_clouds"].append(low_cloud)
        data["mid_clouds"].append(mid_cloud)
        data["high_clouds"].append(high_cloud)
        data["vertical_visibility"].append(vertical_visibility)

    df = pd.DataFrame(data)

    # Convert 'time' to datetime if it's not already
    # Convert 'time' column to datetime, coercing errors to NaT
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Drop rows with NaT values in the 'time' column
    df.dropna(subset=['time'], inplace=True)

    # Convert 'temperature' and 'dewpoint' to numeric
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['dewpoint'] = pd.to_numeric(df['dewpoint'], errors='coerce')

    # Apply smoothing using a moving average
    window_size = 3
    df['temperature_smoothed'] = df['temperature'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['dewpoint_smoothed'] = df['dewpoint'].rolling(window=window_size, center=True, min_periods=1).mean()

    # Interpolate to fill any remaining NaN values after smoothing
    df['temperature_smoothed'] = df['temperature_smoothed'].interpolate(method='linear')
    df['dewpoint_smoothed'] = df['dewpoint_smoothed'].interpolate(method='linear')

    # Drop any rows where interpolation failed to fill NaNs (optional, but safer)
    df.dropna(subset=['temperature_smoothed', 'dewpoint_smoothed'], inplace=True)

    # Debugging prints to trace data flow
    print(df[['temperature', 'dewpoint']])  # Check the DataFrame before further processing

    return df

def to_heat_index(tempF, dewF):
    """
    Calculate the heat index given temperature and dew point in Fahrenheit.
    Returns the heat index in Fahrenheit.
    """
    # Make sure both inputs are numpy arrays to work element-wise
    tempF = np.array(tempF)
    dewF = np.array(dewF)

    # Formula for calculating heat index
    if np.any(tempF < 80):
        return np.nan  # Heat index is only valid for temperatures >= 80°F

    # Constants for the heat index formula
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 * (10 ** -3)
    c6 = -5.481717 * (10 ** -2)
    c7 = 1.22874 * (10 ** -3)
    c8 = 8.5282 * (10 ** -4)
    c9 = -1.99 * (10 ** -6)

    rh = 100 * (112 - dewF) / (112 - tempF)  # Rough approximation for relative humidity

    heat_index = (c1 + (c2 * tempF) + (c3 * rh) + (c4 * tempF * rh) +
                  (c5 * tempF ** 2) + (c6 * rh ** 2) + (c7 * tempF ** 2 * rh) +
                  (c8 * tempF * rh ** 2) + (c9 * tempF ** 2 * rh ** 2))

    return heat_index

def to_wind_chill(tempF, wind_speed):
    """
    Calculate the wind chill given temperature (in Fahrenheit) and wind speed (in knots).
    Wind chill is valid when the temperature is at or below 50°F and wind speed is above 5 knots.
    Returns the wind chill in Fahrenheit.
    """
    # Convert wind speed from knots to mph
    wind_speed_mph = wind_speed * 1.15078  # 1 knot = 1.15078 mph

    # Only apply wind chill formula if tempF <= 50°F and wind_speed > 5 knots
    wind_chill = np.where((tempF <= 50) & (wind_speed > 5),
                          35.74 + 0.6215 * tempF - 35.75 * wind_speed_mph**0.16 + 0.4275 * tempF * wind_speed_mph**0.16,
                          np.nan)  # Return NaN if conditions aren't met

    return wind_chill

@bot.command()
async def meteogram(ctx, icao: str, hoursback: int):
    try:
        metar_list = get_metar(icao, hoursback)
        df = process_metar_data(metar_list)

        # Get current UTC time
        utc_time = datetime.now(pytz.utc)

        # Check if the DataFrame is empty before plotting
        if df.empty:
            await ctx.send("No valid METAR data available for plotting.")
            return

        # Convert temperatures to Fahrenheit
        df['tempF'] = (df['temperature'] * 9 / 5) + 32
        df['dewF'] = (df['dewpoint'] * 9 / 5) + 32

        # Calculate heat index, wind chill, etc.
        df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)
        df.loc[df['tempF'] < 80, 'heat_index'] = np.nan
        df['wind_chill'] = to_wind_chill(df['tempF'].values, df['wind_speed'].values)
        df.loc[(df['wind_speed'] <= 5) | (df['tempF'] > 50), 'wind_chill'] = np.nan

        # Wet bulb temperature calculation with error handling
        try:
            df['wet_bulb'] = mpcalc.wet_bulb_temperature(
                df['pressure'].values * units('hPa'),
                df['temperature'].values * units('degC'),
                df['dewpoint'].values * units('degC')
            ).to('degF').m
        except ValueError:
            df['wet_bulb'] = np.nan  # Handle failure to calculate

        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharex=True)
        fig.patch.set_facecolor('lightsteelblue')

        # Set grid for each axis
        for ax in axs:
            ax.grid(True)

        # Set title for the top subplot (axs[0])
        axs[0].set_title(f'Meteogram for {icao.upper()} - Last {hoursback} hours (Generated at: {utc_time.strftime("%Y-%m-%d %H:%M UTC")})', weight='bold', size='16')

        # 1. Temperature, Dewpoint, Wet Bulb, Heat Index, Wind Chill
        if not df[['tempF', 'dewF']].isnull().all().any():
            plt.sca(axs[0])  # Set the current axes to axs[0]
            plt.plot(df['time'], df['tempF'], label='Temperature (°F)', linewidth=3, color='tab:red')
            plt.plot(df['time'], df['dewF'], label='Dewpoint (°F)', linewidth=3, color='tab:green')
            plt.plot(df['time'], df['wet_bulb'], label='Wet Bulb (°F)', linewidth=3, linestyle='dotted', color='tab:blue')
            plt.plot(df['time'], df['heat_index'], label='Heat Index (°F)', linestyle='--', color='tab:orange')
            plt.plot(df['time'], df['wind_chill'], label='Wind Chill (°F)', linestyle='--', color='tab:purple')

            axs[0].set_ylabel('Temperature (°F)')
            axs[0].legend(loc='upper left', fontsize=14, frameon=True, title='Temperature / Dewpoint', prop={'size':10})
        else:
            axs[0].axhline(0, color='black')  # x-axis line
            axs[0].set_ylabel('Temperature (°F)')

        # 2. Wind Speed and Gusts
        valid_wind_data = df.dropna(subset=['wind_direction', 'wind_speed', 'wind_gusts'], how='all')
        if not valid_wind_data.empty:
            scatter = axs[1].scatter(valid_wind_data['time'], valid_wind_data['wind_direction'],
                                     c=valid_wind_data['wind_speed'], cmap='gnuplot', label='Wind Speed (knots)', vmin=0, vmax=100)
            axs[1].scatter(valid_wind_data['time'], valid_wind_data['wind_direction'],
                           c=valid_wind_data['wind_gusts'], cmap='spring', marker='x', label='Wind Gusts (knots)', vmin=0, vmax=100)

            cbar = plt.colorbar(scatter, ax=axs[1], pad=0.002)
            cbar.set_label('Wind Speed (knots)')
            axs[1].set_yticks([0, 90, 180, 270, 360])
            axs[1].set_yticklabels(['N', 'E', 'S', 'W', 'N'])
            axs[1].set_ylim(0, 360)
            axs[1].set_ylabel('Wind Speed (kts)')
            axs[1].legend(loc='upper left', fontsize=14, frameon=True, title='Wind speed & Direction', prop={'size':10})
        else:
            axs[1].axhline(0, color='black')  # x-axis line
            axs[1].set_ylabel('Wind Speed (kts)')

        # 3. Pressure
        if not df['pressure'].isnull().all():
            axs[2].plot(df['time'], df['pressure'], label='Pressure (inHg)', linewidth=3, color='tab:purple')
            axs[2].set_ylabel('Pressure (inHg)')
            axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[2].legend(loc='upper left', fontsize=14, frameon=True, title='Pressure (inHg)', prop={'size':10})
        else:
            axs[2].axhline(0, color='black')  # x-axis line
            axs[2].set_ylabel('Pressure (inHg)')

        # 4. Cloud Cover and Vertical Visibility as Scatter Plot
        axs[3].scatter(df['time'], df['low_clouds'], color='green', label='Low Clouds', marker='s')
        axs[3].scatter(df['time'], df['mid_clouds'], color='blue', label='Mid Clouds', marker='o')
        axs[3].scatter(df['time'], df['high_clouds'], color='red', label='High Clouds', marker='^')
        axs[3].scatter(df['time'], df['vertical_visibility'], color='yellow', label='Vertical Visibility', marker='v')

        axs[3].set_ylabel('Cloud Cover / Vertical Visibility (ft)')
        axs[3].legend(loc='upper left', fontsize=14, frameon=True, title='Cloud Cover', prop={'size':10})
        axs[3].set_ylim(0)  # NO CLOUDS UNDERGROUND DAMMIT

        # Further control x-axis ticks and labels formatting
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Major ticks every 12 hours
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))   # Minor ticks every 3 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))  # Set date format to Day and Hour

        # Ensure x-axis tick labels are shown on all subplots
        for ax in axs[:-1]:  # Apply to all but the last subplot
            ax.xaxis.set_tick_params(labelbottom=True)  # Force tick labels on

        # Rotate x-axis labels on all subplots and set font size
        plt.setp(axs[-1].get_xticklabels(), rotation=60, fontsize=8)  # Last subplot
        plt.setp([ax.get_xticklabels() for ax in axs[:-1]], rotation=60, fontsize=8)  # Other subplots

         # Refine grid and background handling
        for ax in axs:
            ax.grid(True, zorder=0)  # Draw grid lines first with low zorder

            # Get the y-axis limits
            ymin, ymax = ax.get_ylim()

            # Get y-ticks and color every other y-axis space using axhspan
            y_ticks = ax.get_yticks()
            for i in range(len(y_ticks) - 1):
                if i % 2 == 0:
                    ax.axhspan(y_ticks[i], y_ticks[i+1], color='limegreen', alpha=0.3, zorder=1)  # Background with slightly higher zorder

        # Adjust subplot spacing and figure size
        fig.subplots_adjust(hspace=0.4, left=0.05, right=0.97, top=0.95, bottom=0.1)

        # METOC Logo
        logo_img = plt.imread('/home/evanl/Documents/photo.jpg')  # Replace with your logo's actual path
        imagebox = OffsetImage(logo_img, zoom=0.25)  # Adjust 'zoom' to control logo size
        ab = AnnotationBbox(imagebox, (0.98, 3.25), xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
        ax.add_artist(ab)
        # UGA Logo
        usmc_img = plt.imread('/home/evanl/Documents/uga_logo.png')  # Replace with your logo's actual path
        imagebox = OffsetImage(usmc_img, zoom=0.075)  # Adjust 'zoom' to control logo size
        abx = AnnotationBbox(imagebox, (0.995, 2.45), xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
        ax.add_artist(abx)

        #plt.tight_layout()

        # Save and send the image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        await ctx.send(file=discord.File(img_buf, filename=f'meteogram_{icao}.png'))
        plt.close(fig)

    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

# --- Wind Rose Command ---
@bot.command()
async def windrose(ctx, longitude: float, latitude: float, start_date: str, end_date: str):
    """
    Generates a wind rose polar chart for the specified location and time period.

    Args:
        longitude: Longitude of the location.
        latitude: Latitude of the location.
        start_date: Start date in YYYYMMDD format.
        end_date: End date in YYYYMMDD format.
    """

    await ctx.send("Fetching wind data and generating chart. Please wait...")

    # API URL
    api_url = f"https://power.larc.nasa.gov/api/application/windrose/point?Longitude={longitude}&latitude={latitude}&start={start_date}&end={end_date}&format=JSON"

    # Fetch data
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()

                # Extract data for plotting (adjust based on API response structure)
                directions = data['properties']['parameter']['wind_direction']['bins']
                speeds = data['properties']['parameter']['wind_speed']['bins']
                frequencies = data['properties']['parameter']['frequency']

                # Create polar chart
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='polar')

                # Plot data (you might need to adjust based on API response structure)
                for i in range(len(directions)):
                    ax.bar(directions[i] * np.pi / 180, frequencies[:, i],
                           width=np.pi / len(directions), bottom=0.0,
                           label=f'{speeds[i]} m/s')

                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                plt.legend()
                plt.title(f'Wind Rose for ({latitude}, {longitude}) from {start_date} to {end_date}')

                # Save and send chart
                plt.savefig('windrose.png')
                await ctx.send(file=discord.File('windrose.png'))
                plt.close()
            else:
                await ctx.send("Error fetching data from NASA POWER API. Please check your input and try again.")

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
from mpl_toolkits.basemap import Basemap

def plot_forecast_realtime(storm):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initialize Basemap
    m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=50, llcrnrlon=-110, urcrnrlon=-60, resolution='i', ax=ax)

    # Extract latitude and longitude from the storm's current position
    current_position = storm.get_current_position()
    latitudes = [current_position['lat']]
    longitudes = [current_position['lon']]

    # Example data for future forecast points (replace with actual storm data)
    times = np.arange(0, 12, 1)
    latitudes += latitudes[0] + np.random.uniform(-0.5, 0.5, len(times))
    longitudes += longitudes[0] + np.random.uniform(-0.5, 0.5, len(times))

    x, y = m(longitudes, latitudes)

    # Hide top and right spines
    m.ax.spines['top'].set_visible(False)
    m.ax.spines['right'].set_visible(False)

    # Set only the bottom and left labels
    m.ax.xaxis.set_tick_params(labelbottom=True, labeltop=False)
    m.ax.yaxis.set_tick_params(labelleft=True, labelright=False)

    # Plot forecast track
    m.plot(x, y, marker='o', linestyle='-', color='blue', label='Forecast Track')

    # Add custom title
    plt.title('Forecast Track of ' + storm.name, fontsize=16, fontweight='bold', pad=40)

    # Draw coastlines, countries, etc.
    m.drawcoastlines()
    m.drawcountries()

    # Remove the top and right ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Disable labels on top and right sides
    ax.tick_params(labeltop=False, labelright=False)

    # Use an absolute path for saving the figure
    forecast_image_path = '/home/evanl/Documents/forecast_basemap.png'
    plt.savefig(forecast_image_path)
    plt.close()

    return forecast_image_path

def plot_models(storm):

    # Use an absolute path for saving the figure
    models_image_path = '/home/evanl/Documents/models.png'  # Specify the file path
    plt.savefig(models_image_path)
    plt.close()
    return models_image_path  # Return the file path

def plot_ensembles(storm):
    plt.figure(figsize=(10, 8))

    # Example data (replace with actual ensemble data)
    ensemble_times = np.arange(0, 12, 1)
    ensemble_forecasts = np.random.uniform(low=storm.vmax * 0.5, high=storm.vmax * 1.5, size=(10, len(ensemble_times)))

    for i in range(ensemble_forecasts.shape[0]):
        plt.plot(ensemble_times, ensemble_forecasts[i], alpha=0.3, label='Ensemble ' + str(i + 1))



    # Use an absolute path for saving the figure
    ensembles_image_path = '/home/evanl/Documents/ensembles.png'  # Specify the file path
    plt.savefig(ensembles_image_path)
    plt.close()
    return ensembles_image_path  # Return the file path

from tropycal.realtime import Realtime
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For custom colorbar positioning

@bot.command()
async def hurricane(ctx, storm_id: str):
    """Displays information about a specific hurricane or potential tropical cyclone."""

    try:
        realtime_obj = Realtime()
        storm = realtime_obj.get_storm(storm_id.upper())

        if storm:
            embed = discord.Embed(title=storm.name, color=discord.Color.blue())

            # Accessing category information
            category = storm.attrs.get('category', "Not available") if storm.invest else "Not available"
            embed.add_field(name="Category", value=category, inline=True)

            # Basic storm information
            embed.add_field(name="Location", value=f"{storm.lat}, {storm.lon}", inline=True)
            embed.add_field(name="Wind Speed", value=f"{storm.vmax} mph", inline=True)
            embed.add_field(name="Pressure", value=f"{storm.mslp} mb", inline=True)

            # Generate and send forecast plot
            try:
                fig, ax = plt.subplots(figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})
                storm.plot_forecast_realtime(ax=ax)  # Generate the plot

                # Add a white patch to cover the Tropycal-generated title
                fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color='white', zorder=5))

                # Add custom title above the plot
                fig.suptitle(f"Hurricane {storm.name} Forecast", fontsize=16, y=0.98)

                plt.tight_layout()

                # Save the forecast plot
                forecast_image_path = '/home/evanl/Documents/forecast.png'
                plt.savefig(forecast_image_path, bbox_inches='tight')
                plt.close()
                with open(forecast_image_path, 'rb') as f:
                    file = discord.File(f, filename='forecast.png')
                    embed.set_image(url='attachment://forecast.png')
                await ctx.send(file=file, embed=embed)
            except Exception as e:
                print(f"Error generating or sending forecast plot: {e}")
                await ctx.send("An error occurred while generating the forecast plot.")

            # Plot models (no colorbar needed here)
            try:
                fig, ax = plt.subplots(figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})
                storm.plot_models(ax=ax)

                # Add a white patch to cover the Tropycal-generated title
                fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color='white', zorder=5))

                # Add custom title above the plot
                fig.suptitle(f"Model Forecast Tracks for {storm.name}", fontsize=16, y=0.98)

                plt.tight_layout()

                # Save the models plot
                models_image_path = '/home/evanl/Documents/models.png'
                plt.savefig(models_image_path, bbox_inches='tight')
                plt.close()
                with open(models_image_path, 'rb') as f:
                    file = discord.File(f, filename='models.png')
                    await ctx.send(file=file, embed=discord.Embed(title="Models Plot", color=discord.Color.blue()))
            except Exception as e:
                print(f"Error generating or sending models plot: {e}")
                await ctx.send("An error occurred while generating the models plot.")

            # Plot ensembles (adjust colorbar)
            try:
                # Create a standard Matplotlib figure and axis without geographic projection
                fig, ax = plt.subplots(figsize=(14, 13))  # No 'projection' argument here since it's not needed
                storm.plot_ensembles(ax=ax)  # Use plot_ensembles directly

                # Add a white patch to cover the Tropycal-generated title
                fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color='white', zorder=5))

                # Add custom title above the plot
                fig.suptitle(f"GEFS Forecast Tracks for {storm.name}", fontsize=16, y=0.98)

                # Use tight layout but keep space for the colorbar
                plt.tight_layout()

                # Adjust the colorbar by shifting it to the right using a divider for colorbar placement
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust the colorbar size and position
                plt.colorbar(ax.collections[-1], cax=cax)  # Attach colorbar to the side

                # Save the ensembles plot
                ensembles_image_path = '/home/evanl/Documents/ensembles.png'
                plt.savefig(ensembles_image_path, bbox_inches='tight')
                plt.close()

                with open(ensembles_image_path, 'rb') as f:
                    file = discord.File(f, filename='ensembles.png')
                    await ctx.send(file=file, embed=discord.Embed(title="Ensembles Plot", color=discord.Color.blue()))

            except Exception as e:
                print(f"Error generating or sending ensembles plot: {e}")
                await ctx.send("An error occurred while generating the ensembles plot.")

        else:
            await ctx.send(f"Storm with ID '{storm_id}' not found.")

    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("An error occurred while fetching storm data. Please try again later.")

@bot.command()
async def active_storms(ctx):
    """Lists all active storms in the North Atlantic basin and produces forecast images."""
    try:
        realtime_obj = realtime.Realtime()
        active_storms = realtime_obj.list_active_storms(basin='north_atlantic')

        if active_storms:
            embed = discord.Embed(title="Active Storms in the North Atlantic", color=discord.Color.blue())

            for storm_id in active_storms:
                storm = realtime_obj.get_storm(storm_id)

                # Basic storm information for the embed
                storm_name = storm.name
                storm_lat = storm.lat
                storm_lon = storm.lon
                storm_vmax = storm.vmax
                storm_mslp = storm.mslp

                storm_info = f"{storm_name} ({storm_id}): {storm_vmax} mph, {storm_mslp} mb"
                embed.add_field(name=storm_name, value=storm_info, inline=False)

                # Generate forecast plot for each storm
                try:
                    storm.plot_forecast_realtime()
                    forecast_image_path = f'/home/evanl/Documents/{storm_id}_forecast.png'
                    plt.savefig(forecast_image_path)
                    plt.close()

                    # Send forecast plot image
                    with open(forecast_image_path, 'rb') as f:
                        file = discord.File(f, filename=f'{storm_id}_forecast.png')
                        embed.set_image(url=f'attachment://{storm_id}_forecast.png')
                        await ctx.send(file=file, embed=embed)
                except Exception as e:
                    print(f"Error generating or sending forecast plot for storm {storm_id}: {e}")
                    await ctx.send(f"An error occurred while generating the forecast plot for {storm_name}.")
        else:
            await ctx.send("No active storms in the North Atlantic at the moment.")

    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("An error occurred while fetching active storm data. Please try again later.")

@bot.command()
async def rainfall(ctx, storm_name: str, year: int):
    """Displays tropical cyclone rainfall data for a historical storm."""

    try:
        # Load dataset for the desired basin (e.g., North Atlantic)
        basin = tracks.TrackDataset(basin='north_atlantic')

        # Get the storm from the dataset
        storm = basin.get_storm((storm_name, year))

        if storm:
            embed = discord.Embed(title=f"{storm_name} ({year}) - Rainfall Data", color=discord.Color.blue())

            # Attempt to generate and send rainfall plot
            try:
                rainfall_image_path = f'/home/evanl/Documents/{storm_name}_{year}_rainfall.png'

                # Plot the rainfall using Tropycal's method
                storm.plot_TC_rain(save_path=rainfall_image_path)

                # Send the image to Discord
                with open(rainfall_image_path, 'rb') as f:
                    file = discord.File(f, filename=f'{storm_name}_{year}_rainfall.png')
                    embed.set_image(url=f'attachment://{storm_name}_{year}_rainfall.png')
                    await ctx.send(file=file, embed=embed)

            except Exception as e:
                print(f"Error generating or sending rainfall plot: {e}")
                await ctx.send("An error occurred while generating the rainfall plot.")

        else:
            await ctx.send(f"Storm '{storm_name}' from {year} not found in the dataset.")

    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("An error occurred while fetching storm data. Please try again later.")

# Initialize Realtime object
realtime_obj = realtime.Realtime()

# Define geographical bounds for different basins
REGION_BOUNDS = {
    'atlantic': (-100, -10, 0, 60),  # Atlantic Ocean (longitude, latitude range)
    'north_pac': (120, -80, 0, 60),  # North Pacific Ocean
    'south_pac': (120, -80, -60, 0),  # South Pacific Ocean
}

def filter_storms_by_region(storms, bounds):
    """Filter storms by region using geographical bounds."""
    lon_min, lon_max, lat_min, lat_max = bounds

    filtered_storms = []
    for storm_id in storms:
        # Fetch the full storm object using its identifier
        storm = realtime_obj.get_storm(storm_id)

        # Access the storm's longitude and latitude attributes
        storm_lon = storm.lon
        storm_lat = storm.lat

        # Check if the storm is within the region bounds
        if (lon_min <= storm_lon[-1] <= lon_max) and (lat_min <= storm_lat[-1] <= lat_max):
            filtered_storms.append(storm)

    return filtered_storms

from PIL import Image

@bot.command()
async def realtime(ctx, region: str = 'global'):
    """Displays a summary of active tropical systems for a specified region or worldwide."""

    try:
        # Normalize the region to lowercase
        region_lower = region.lower()

        # Create a wider figure but maintain the height using Matplotlib and Cartopy
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})  # Wide figure size

        # Set background color to lightsteelblue
        fig.patch.set_facecolor('lightsteelblue')

        # Ensure there's no previous title
        ax.set_title('')  # Clear any existing title before adding the custom one

        if region_lower == 'global':
            print("Generating worldwide plot")
            realtime_obj.plot_summary(ax=ax)  # Pass the manually created axis

        elif region_lower in REGION_BOUNDS:
            lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region_lower]
            print(f"Generating plot for {region_lower} with domain: lon({lon_min}, {lon_max}), lat({lat_min}, {lat_max})")
            realtime_obj.plot_summary(domain={'w': lon_min, 'e': lon_max, 's': lat_min, 'n': lat_max}, ax=ax)

        else:
            await ctx.send("Invalid region or unsupported region. Supported regions: atlantic, north_pac, south_pac.")
            return

        # Adjust the layout to make the plot area fit the entire figure width
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.10)  # Adjust to stretch the plot

        # Force the aspect ratio to auto to fill the figure
        ax.set_aspect('auto')

        # Set the title and customize its position, ensuring it's added only once
        ax.set_title(f"Summary of Active Systems in {region.capitalize()}", fontsize=14, pad=20)

        # Remove axis labels and unnecessary spines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Load and resize the icons, multiplying the height scaling by 3 for a larger size
        dpi = fig.dpi
        quarter_inch_in_pixels = 0.25 * dpi * 4  # 3x larger than original quarter inch height

        uga_logo = Image.open('/home/evanl/Documents/uga_logo.png')
        photo = Image.open('/home/evanl/Documents/photo.jpg')

        uga_logo_resized = uga_logo.resize((int(quarter_inch_in_pixels * uga_logo.width / uga_logo.height), int(quarter_inch_in_pixels)))
        photo_resized = photo.resize((int(quarter_inch_in_pixels * photo.width / photo.height), int(quarter_inch_in_pixels)))

        # Add the resized icons to the plot
        ax.figure.figimage(uga_logo_resized, 10, fig.bbox.ymax - uga_logo_resized.height - 10, zorder=1)
        ax.figure.figimage(photo_resized, fig.bbox.xmax - photo_resized.width - 10, fig.bbox.ymax - photo_resized.height - 10, zorder=1)

        # Save and send the plot
        summary_image_path = os.path.join('/home/evanl/Documents', f'summary_{uuid4()}.png')
        plt.savefig(summary_image_path)
        plt.close()

        # Send the plot in Discord
        with open(summary_image_path, 'rb') as f:
            file = discord.File(f, filename='summary.png')
            embed = discord.Embed(title=f"Summary of Active Systems in {region.capitalize()}", color=discord.Color.blue())
            embed.set_image(url='attachment://summary.png')
            await ctx.send(file=file, embed=embed)

    except Exception as e:
        print(f"Error generating summary plot: {e}")
        await ctx.send("An error occurred while generating the summary plot.")

def get_taf_checkwx(icao):
    """
    Fetches TAF from the CheckWX API for the specified ICAO code.
    """

    api_key = "c0eead33ce0a4403800f26f173"
    base_url = "https://api.checkwx.com/taf/"
    url = f"{base_url}{icao}/decoded"

    headers = {
        "X-API-Key": api_key
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        # Check if TAF data is available
        if "data" in data and len(data["data"]) > 0:
            taf_data = data["data"][0]  # Assuming the first TAF is the most recent
            raw_taf = taf_data["raw_text"]
            return raw_taf
        else:
            raise ValueError(f"No TAF data found for {icao}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching TAF data for {icao} from CheckWX: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing TAF data for {icao} from CheckWX: {e}")

@bot.command()
async def taf(ctx, airport_code: str):
    """
    Sends the latest TAF for the given airport, splitting at BECMG/FM lines and using an embed.
    """
    airport_code = airport_code.upper()

    try:
        raw_taf = get_taf_checkwx(airport_code)

        # Split the TAF into sections based on BECMG or FM
        taf_sections = re.split(r'(BECMG|FM)', raw_taf)

        embed = discord.Embed(title=f"TAF for {airport_code}", color=0xc0db01)

        current_section_name = "Initial"  # Start with "Initial"

        # Add each section as a field in the embed
        for i, section in enumerate(taf_sections):
            if section.strip():  # Skip empty sections
                if section.startswith(('BECMG', 'FM')):
                    current_section_name = section.strip()  # Update the section name
                else:
                    embed.add_field(name=current_section_name, value=f"`\n{section.strip()}\n`", inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        # Handle any errors gracefully and inform the user
        await ctx.send(f"Sorry, there was an error fetching the TAF for {airport_code}: {e}")
        logging.error(f"Error fetching TAF for {airport_code}: {e}")

# --- Maps Command ---


# --- Sigmets & Airmets Command --- 
def get_aviation_weather_alerts_checkwx(icao):
    """
    Fetches SIGMETs and AIRMETs from the CheckWX API for the specified ICAO code, handling potential string responses.
    """

    api_key = "c0eead33ce0a4403800f26f173" 
    base_url = "https://api.checkwx.com/"

    try:
        alerts = []
        for alert_type, endpoint in [("AIRMET", "airmet"), ("SIGMET", "sigmet")]:
            url = f"{base_url}{endpoint}/{icao}"
            response = requests.get(url, headers={"X-API-Key": api_key})
            response.raise_for_status()
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                for alert in data["data"]:
                    # Handle potential string response
                    if isinstance(alert, str):
                        alerts.append({
                            "type": alert_type,
                            "raw_text": alert,
                            "valid_from": "Not available",
                            "valid_to": "Not available"
                        })
                    else:  # Assume dictionary format if not a string
                        alerts.append({
                            "type": alert_type,
                            "raw_text": alert.get("raw_text", "Raw text not available"),
                            "valid_from": alert.get("valid_from", "Valid from not available"),
                            "valid_to": alert.get("valid_to", "Valid to not available")
                        })

        return alerts

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching aviation weather alerts for {icao} from CheckWX: {e}")

@bot.command()
async def mets(ctx, airport_code: str):
    """
    Sends the latest AIRMETs and SIGMETs for the given airport.
    """
    airport_code = airport_code.upper()

    try:
        alerts = get_aviation_weather_alerts_checkwx(airport_code)

        if not alerts:
            await ctx.send(f"No AIRMETs or SIGMETs found for {airport_code}.")
            return

        for alert in alerts:
            embed = discord.Embed(title=f"{alert['type']} for {airport_code}",
                                  description=f"`\n{alert['raw_text']}\n`",
                                  color=0xa41500)  # red color
            embed.add_field(name="Valid From", value=alert['valid_from'], inline=True)
            embed.add_field(name="Valid To", value=alert['valid_to'], inline=True)
            await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"Sorry, there was an error fetching aviation weather alerts for {airport_code}: {e}")
        logging.error(f"Error fetching aviation weather alerts for {airport_code}: {e}")

# --- Imported SkewT Command ---
bot.add_command(skewt)

# --- Satellite Command ---
@bot.command()
async def sat(ctx, region: str, product_code: int):
    """Fetches satellite image for the specified region and product code using pre-defined links."""

    try:
        region = region.lower()
        valid_regions = ["conus", "fulldisk", "mesosector", "mesosector2", "tropicalatlantic", "gomex", "ne", "indian", "capeverde", "neatl", 'fl', 'pacus', 'wc', 'ak', 'wmesosector', 'wmesosector2']

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")

        # Product codes for different regions (updated with new regions and product codes, and GeoColor removed)
        product_codes = {
            "conus": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "fulldisk": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 5: "RGB Air Mass"},
            "mesosector": {2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "mesosector2": {2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "tropicalatlantic": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
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

import airportsdata

@bot.command()
async def astro(ctx, icao: str = None):  # Get ICAO code directly from command
    """Provides sunrise, sunset, moon phase, and twilight information for a given location."""
    if not icao:
        await ctx.send("Please provide an ICAO code (e.g., '$astro kmge')")
        return

    try:
        # Get airport data using ICAO code
        airports = airportsdata.load('icao')  # Load airport data by ICAO code
        airport = airports.get(icao.upper())  # Get airport data
        if not airport:
            raise ValueError("Airport not found.")

        # Extract latitude and longitude
        lat = airport['lat']
        lon = airport['lon']

        # Determine time zone (use lat, lon from airport data)
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lng=lon, lat=lat)

        # Get current time in the specified time zone
        now = datetime.now(pytz.timezone(timezone))

        # Calculate sunrise, sunset, and twilight times using PyEphem
        obs = ephem.Observer()
        obs.lat = str(lat)  # Use lat from airport data
        obs.lon = str(lon)  # Use lon from airport data
        obs.date = now

        sun = ephem.Sun()
        moon = ephem.Moon()

        # Sunrise and Sunset Calculations
        sunrise = ephem.localtime(obs.next_rising(sun))
        sunset = ephem.localtime(obs.next_setting(sun))

        # Twilight Calculations
        obs.horizon = '-0:34'  # Civil twilight
        civil_twilight_begin = ephem.localtime(obs.previous_rising(sun, use_center=True))
        civil_twilight_end = ephem.localtime(obs.next_setting(sun, use_center=True))

        obs.horizon = '-6'  # Nautical twilight
        nautical_twilight_begin = ephem.localtime(obs.previous_rising(sun, use_center=True))
        nautical_twilight_end = ephem.localtime(obs.next_setting(sun, use_center=True))

        obs.horizon = '-12'  # Astronomical twilight
        astronomical_twilight_begin = ephem.localtime(obs.previous_rising(sun, use_center=True))
        astronomical_twilight_end = ephem.localtime(obs.next_setting(sun, use_center=True))

        # Calculate moon phase
        moon.compute(obs.date)
        moon_phase = moon.phase

        # Create the embed (use icao here)
        embed = discord.Embed(title=f"**Astronomy information for {icao.upper()}**", color=0x808080)
        embed.add_field(name="**Sunrise**", value=sunrise.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Sunset**", value=sunset.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Civil Twilight Begin**", value=civil_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Civil Twilight End**", value=civil_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Nautical Twilight Begin**", value=nautical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Nautical Twilight End**", value=nautical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Astronomical Twilight Begin**", value=astronomical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Astronomical Twilight End**", value=astronomical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z'), inline=False)
        embed.add_field(name="**Moon Phase**", value=f"{moon_phase:.1f}% (Illuminated)", inline=False)

        # Generate Sun Path Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        times = [now + timedelta(minutes=15 * i) for i in range(96)]  # 15-minute intervals over 24 hours

        # Data for plotting segments
        sun_azimuth = []
        sun_altitude = []
        segments = []

        current_segment = []

        # Calculate sun positions for each time interval
        for time in times:
            obs.date = time
            sun.compute(obs)

            azimuth = np.degrees(sun.az)
            altitude = np.degrees(sun.alt)

            # If the sun goes below the horizon, end the current segment
            if current_segment and (altitude < 0) != (current_segment[-1][1] < 0):
                segments.append(current_segment)
                current_segment = []

            # Avoid appending the same altitude value repeatedly to prevent flat lines
            if current_segment and current_segment[-1][1] == altitude:
                continue

            # NEW CONDITION: Also avoid appending if the azimuth has wrapped around
            if current_segment and abs(current_segment[-1][0] - azimuth) > 180:
                segments.append(current_segment)
                current_segment = []

            current_segment.append((azimuth, altitude))


        # Append the last segment if not empty
        if current_segment:
            segments.append(current_segment)

        # Plot sun path segments
        for segment in segments:
            azimuths, altitudes = zip(*segment)
            linestyle = '-' if altitudes[0] >= 0 else '--'
            ax.plot(azimuths, altitudes, color='orange', lw=3, linestyle=linestyle)

        # Calculate current sun and moon positions
        obs.date = now
        sun.compute(obs)
        moon.compute(obs)

        current_sun_az = np.degrees(sun.az)
        current_sun_alt = np.degrees(sun.alt)

        current_moon_az = np.degrees(moon.az)
        current_moon_alt = np.degrees(moon.alt)

        # Plot twilight periods along y-axis
        ax.axhspan(-90, -12, color='midnightblue', alpha=0.3, label='Astronomical Twilight')
        ax.axhspan(-12, -6, color='deepskyblue', alpha=0.3, label='Nautical Twilight')
        ax.axhspan(-6, -0.34, color='lightskyblue', alpha=0.3, label='Civil Twilight')
        # Shade nighttime
        ax.axhspan(-90, -18, color='black', alpha=0.5, label='Night')

        # Mark the current position of the sun
        ax.scatter(current_sun_az, current_sun_alt, color='yellow', edgecolors='black', s=150, label='Current Sun Position', zorder=3)

        # Mark the current position of the moon
        ax.scatter(current_moon_az, current_moon_alt, color='gray', edgecolors='black', s=150, label='Current Moon Position', zorder=3)

        # Horizon Line
        ax.axhline(0, color='blue', linestyle='--', lw=0.8, label='Horizon')  # Horizon line

        # Labels and title
        # Get ICAO code (if available)
        #icao = loc.raw.get('icao', 'N/A')  # Get ICAO code, default to 'N/A' if not found
        # Labels and title
        # Labels and title (use icao directly)
        # Labels and title
        ax.set_title(f"Sun/Moon Position: {icao.upper()} - {lat:.4f}, {lon:.4f}\n{now.strftime('%b %d %Y %H:%M')} Local Time")
        ax.set_xlabel("Azimuth (degrees)")
        ax.set_ylabel("Altitude (degrees)")
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

        # Customize x-axis to show cardinal directions
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(['N', 'E', 'S', 'W', 'N'])

        # Legend and grid
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # Save plot to a file
        plot_filename = "/home/evanl/Documents/sun_moon_plot.png"  # Adjust the path to an existing directory
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig)

        # Send the embed and the plot image
        await ctx.send(embed=embed)
        await ctx.send(file=discord.File(plot_filename))

        logging.info(f"User {ctx.author} requested astronomy information for {location}")

    except (AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving astronomy information: {e}")

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
	    
# --- ASCAT Command ---
# Constants
BASE_URL_NHC = "https://www.nhc.noaa.gov"

@bot.command()
async def ascat(ctx):
    """Lists currently active storms from the NHC."""

    try:
        # Fetch the NHC Atlantic basin page (adjust if needed for other basins)
        url_nhc = f"{BASE_URL_NHC}/text/refresh/MIATCPAT1+shtml/051835.shtml"
        response_nhc = requests.get(url_nhc)
        response_nhc.raise_for_status()

        # Parse the NHC page to find active storms
        ascat = get_active_storms_from_nhc(response_nhc.content)

        if ascat:
            storm_list = [f"{s['id']} - {s['name']} ({s['basin']})" for s in ascat]
            await ctx.send(f"Currently active storms (from NHC):\n{', '.join(storm_list)}")
        else:
            await ctx.send("No active storms found according to NHC. Please check again later.")

    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error connecting to the NHC website: {e}")
    except AttributeError as e:
        await ctx.send(f"Error parsing the NHC webpage. The website structure might have changed: {e}")


# Functions for parsing

def get_active_storms_from_nhc(html_content):
    """Parses NHC HTML to extract active storms."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the 'pre' tag containing the storm information
    pre_tag = soup.find('pre')
    if not pre_tag:
        return []  # No storms found or unexpected page structure

    # Extract the relevant lines containing storm data
    lines = pre_tag.text.splitlines()
    storm_lines = [line for line in lines if line.startswith("AL") or line.startswith("EP")]

    storms = []
    for line in storm_lines:
        parts = line.split()
        if len(parts) >= 3:
            storm_id = parts[0]  # e.g., 'AL092023'
            storm_name = ' '.join(parts[1:-1])  # e.g., 'FRANKLIN'
            basin = parts[-1]   # e.g., 'AL' (Atlantic) or 'EP' (Eastern Pacific)
            storms.append({'id': storm_id, 'basin': basin, 'name': storm_name})

    return storms

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
                embed = discord.Embed(title=properties['headline'], color=discord.Color.purple())
                # ... (other embed fields)

                # Clean up and format the area descriptions
                cleaned_area_desc = "".join(properties['areaDesc']).split(";")
                cleaned_area_desc = [area.strip() for area in cleaned_area_desc if area.strip()]
                area_desc = ", ".join(cleaned_area_desc) 
                embed.add_field(name="Area", value=area_desc, inline=False)

                # Add Description field with a check for existence and default value
                description = properties.get('description', "No description available.") 
                embed.add_field(name="Description", value=description, inline=False)

                # ... (rest of the embed fields)
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

@bot.command() 
async def forecast(ctx, icao_code: str):
    api_key = "efd4f5ec6d2b16958a946b2ceb0419a6"
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    complete_url = base_url + "?appid=" + api_key + "&units=imperial" + "&id=" + icao_code

    async with aiohttp.ClientSession() as session: 
        async with session.get(complete_url) as response:
            if response.status == 200:
                data = await response.json()

                # Check if 'list' exists in the response (indicating successful forecast retrieval)
                if 'list' in data:
                    forecast_list = data['list']

                    # Extract and format forecast data for 5 days
                    forecast_data = []
                    for i in range(0, 40, 8):  
                        forecast = forecast_list[i]
                        date_time = forecast['dt_txt']
                        temp = forecast['main']['temp']
                        description = forecast['weather'][0]['description']
                        forecast_data.append(f"{date_time}: {temp}°F, {description}")

                    await ctx.send("\n".join(forecast_data))
                else:
                    await ctx.send("Error fetching forecast data. Please check the ICAO code.") 
            else:
                await ctx.send("Error fetching forecast data.")

    
async def get_lightning_data(lat, lon):
    """
    Fetches current lightning strike data for a given latitude and longitude within a specified radius and timeframe.
    """
    params = {
        'lat': lat,
        'lon': lon,
        'key': API_KEY,
        'search_dist_km': 100,  # Adjust search radius as needed
        'limit': 5,             # Adjust limit as needed
        'search_mins': 15,      # Adjust timeframe as needed
        'sort': 'distance'      # Sort by distance
    }

    try:
        response = requests.get('https://api.weatherbit.io/v2.0/current/lightning', params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching lightning data: {e}")
        return None

async def get_coordinates_from_icao(icao_code):
    """
    Fetches latitude and longitude for a given ICAO code.
    Prioritizes using the Weatherbit API, falls back to a hardcoded dictionary if no data is found from the API.
    """

    # Try fetching from Weatherbit API first
    url = f'https://api.weatherbit.io/v2.0/airports?icao={icao_code}&key={API_KEY}'
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()

        if data and data['count'] > 0:
            airport = data['data'][0] 
            lat = airport['lat']
            lon = airport['lon']
            return lat, lon
    except requests.exceptions.RequestException as e:
        print(f"Error fetching airport data from Weatherbit API: {e}")

    # If Weatherbit API fails or doesn't find the airport, fallback to the hardcoded dictionary
    airport_data = { 
        "KATL": (33.6367, -84.4281),  # Atlanta
        "KJFK": (40.6398, -73.7789),  # New York
        "KBIX": (30.4103, -88.9261),  # Keesler Air Force Base
        "KVQQ": (30.2264, -81.8878),  # Cecil Airport
        "KVPC": (34.1561, -84.7983),  # Cartersville Airport
        "KRMg": (34.4956, -85.2214),  # Richard B Russell Airport
        "KMGE": (33.9131, -84.5197),  # Dobbins Air Reserve Base
        "KGPT": (30.4075, -89.0753),  # Gulfport-Biloxi International Airport
        "KPIT": (40.4915, -80.2329),  # Pittsburgh International Airport
        "KSGJ": (34.25, -84.95),      # St Simons Island Airport
        "KPEZ": (40.3073, -75.6192),   # somewhere in TEXAS
        "KNSE": (30.72247, -87.02390),  # Milton/Whiting Field Naval Air Station North
        "KTPA": (27.9755, -82.5332)   # Tampa International Airport
        # ... add more airports as needed 
    }
    return airport_data.get(icao_code.upper())
    
@bot.command()
async def lightning(ctx, icao_code: str):
    """Gets lightning strike data near an airport specified by ICAO code."""
    coordinates = await get_coordinates_from_icao(icao_code)

    if coordinates:
        lat, lon = coordinates
        lightning_data = await get_lightning_data(lat, lon)

        if lightning_data:
            strikes = lightning_data.get('data', [])

            if strikes:
                # Lightning strikes were found
                embed = discord.Embed(title=f"Lightning Strikes near {icao_code.upper()}", color=discord.Color.blue())

                for strike in strikes[:5]: 
                    embed.add_field(name="Strike Time (UTC)", value=strike['strike_time'], inline=False)
                    embed.add_field(name="Latitude", value=strike['lat'], inline=True)
                    embed.add_field(name="Longitude", value=strike['lon'], inline=True)

                await ctx.send(embed=embed)
            else:
                # No lightning strikes were found
                await ctx.send(f"No lightning strikes detected near {icao_code.upper()}.")
        else:
            # Error fetching lightning data
            await ctx.send("An error occurred while fetching lightning data.")
    else:
        # ICAO code not found
        await ctx.send(f"Airport data not found for ICAO code '{icao_code}'. Please check the code and try again.")

# Import all plot functions from RTMA_Graphics.py
from RTMA_Graphics import (
    plot_relative_humidity, plot_24_hour_relative_humidity_comparison,
    plot_temperature, plot_frost_freeze, plot_extreme_heat,
    plot_24_hour_temperature_comparison, plot_dew_point,
    plot_24_hour_dew_point_comparison, plot_total_cloud_cover,
    plot_24_hour_total_cloud_cover_comparison, plot_wind_speed,
    plot_24_hour_wind_speed_comparison, plot_wind_speed_and_direction,
    plot_24_hour_wind_speed_and_direction_comparison, plot_dry_and_windy_areas,
    plot_dry_and_gusty_areas, plot_relative_humidity_with_metar_obs,
    plot_low_relative_humidity_with_metar_obs
)

# --- Maps Command ---
@bot.command()
async def maps(ctx, map_type: str = 'humidity'):
    """
    Generates a weather map based on the user's selection.

    Supported map types:
    - 'humidity': Relative humidity map.
    - '24_hour_humidity': 24-hour relative humidity comparison.
    - 'temperature': Temperature map.
    - 'frost_freeze': Frost and freeze areas map.
    - 'extreme_heat': Extreme heat areas map.
    - 'dew_point': Dew point map.
    - 'cloud_cover': Total cloud cover map.
    - 'wind_speed': Wind speed map.
    - 'wind_direction': Wind speed and direction map.
    - 'dry_windy': Dry and windy areas map.
    """

    try:
        # Map type selection logic
        if map_type.lower() == 'humidity':
            plot_relative_humidity()
        elif map_type.lower() == '24_hour_humidity':
            plot_24_hour_relative_humidity_comparison()
        elif map_type.lower() == 'temperature':
            plot_temperature()
        elif map_type.lower() == 'frost_freeze':
            plot_frost_freeze()
        elif map_type.lower() == 'extreme_heat':
            plot_extreme_heat()
        elif map_type.lower() == 'dew_point':
            plot_dew_point()
        elif map_type.lower() == 'cloud_cover':
            plot_total_cloud_cover()
        elif map_type.lower() == 'wind_speed':
            plot_wind_speed()
        elif map_type.lower() == 'wind_direction':
            plot_wind_speed_and_direction()
        elif map_type.lower() == 'dry_windy':
            plot_dry_and_windy_areas()
        else:
            raise ValueError(f"Invalid map type '{map_type}'. Please choose a valid option.")

        # Save the plot to a file
        file_name = f'{map_type}_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

        # Send the file to Discord
        with open(file_name, 'rb') as f:
            picture = discord.File(f)
            await ctx.send(f"Here is your {map_type} map:", file=picture)

    except ValueError as e:
        await ctx.send(str(e))
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

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

# --- Convert Commmand ---
# Define a dictionary to store conversion factors
conversion_factors = {
    # Temperature
    "C": {"F": lambda x: (9/5) * x + 32, "K": lambda x: x + 273.15},
    "F": {"C": lambda x: (5/9) * (x - 32), "K": lambda x: (5/9) * (x - 32) + 273.15},
    "K": {"C": lambda x: x - 273.15, "F": lambda x: (9/5) * (x - 273.15) + 32},

   # Length/Distance (Enhanced with small units)
    "M": {"KM": lambda x: x * 0.001, "FT": lambda x: x * 3.28084, "MI": lambda x: x * 0.000621371, "NM": lambda x: x * 0.000539957,
          "CM": lambda x: x * 100, "MM": lambda x: x * 1000, "IN": lambda x: x * 39.3701}, # New additions
    "KM": {"M": lambda x: x * 1000, "FT": lambda x: x * 3280.84, "MI": lambda x: x * 0.621371, "NM": lambda x: x * 0.539957},
    "FT": {"M": lambda x: x * 0.3048, "KM": lambda x: x * 0.0003048, "MI": lambda x: x * 0.000189394, "NM": lambda x: x * 0.000164579,
          "CM": lambda x: x * 30.48, "MM": lambda x: x * 304.8, "IN": lambda x: x * 12}, # New additions
    "MI": {"M": lambda x: x * 1609.34, "KM": lambda x: x * 1.60934, "FT": lambda x: x * 5280, "NM": lambda x: x * 0.868976},
    "NM": {"M": lambda x: x * 1852, "KM": lambda x: x * 1.852, "FT": lambda x: x * 6076.12, "MI": lambda x: x * 1.15078},
    "AU": {"KM": lambda x: x * 149597870.7, "M": lambda x: x * 149597870700},
    "LY": {"KM": lambda x: x * 9.461e+12, "M": lambda x: x * 9.461e+15},
    "PC": {"LY": lambda x: x * 3.26156, "AU": lambda x: x * 206264.806},
    "CM": {"M": lambda x: x / 100, "IN": lambda x: x / 2.54, "MM": lambda x: x * 10}, # New unit and its conversions
    "MM": {"M": lambda x: x / 1000, "IN": lambda x: x / 25.4, "CM": lambda x: x / 10}, # New unit and its conversions
    "IN": {"M": lambda x: x / 39.3701, "CM": lambda x: x * 2.54, "MM": lambda x: x * 25.4}, # New unit and its conversions

    # Area
    "ACRE": {"M^2": lambda x: x * 4046.86, "KM^2": lambda x: x * 0.00404686, "HA": lambda x: x * 0.404686},
    "MI^2": {"KM^2": lambda x: x * 2.58999},

    # Volume
    "GAL_US": {"L": lambda x: x * 3.78541},
    "GAL_IMP": {"L": lambda x: x * 4.54609},
    "FT^3": {"M^3": lambda x: x * 0.0283168},
    "BBL": {"L": lambda x: x * 158.987, "GAL_US": lambda x: x * 42},

    # Mass/Weight
    "LB": {"KG": lambda x: x * 0.453592},
    "OZ": {"G": lambda x: x * 28.3495},
    "TON_US": {"KG": lambda x: x * 907.185},
    "TONNE": {"KG": lambda x: x * 1000},

    # Force
    "N": {"LBF": lambda x: x * 0.224809},
    "W": {"HP": lambda x: x * 0.00134102},
    "HP": {"W": lambda x: x / 0.00134102},  # Added inverse conversion

    # Torque (Requires RPM for conversion)
    "N_M": {
        "LB_FT": lambda x: x * 0.737562
    },
    "LB_FT": {
        "N_M": lambda x: x / 0.737562
    },
    "HP_TO_TORQUE": lambda hp, rpm: (hp * 5252) / rpm,
    "TORQUE_TO_HP": lambda torque, rpm: (torque * rpm) / 5252,

     # Speed
    "KT": {"MPH": lambda x: x * 1.15078, "KPH": lambda x: x * 1.852, "M/S": lambda x: x * 0.514444},
    "MPH": {"KT": lambda x: x / 1.15078, "KPH": lambda x: x * 1.609344, "M/S": lambda x: x * 0.44704},
    "KPH": {"MPH": lambda x: x / 1.609344, "KT": lambda x: x / 1.852, "M/S": lambda x: x * 0.277778},
    "M/S": {"MPH": lambda x: x / 0.44704, "KPH": lambda x: x / 0.277778, "KT": lambda x: x / 0.514444},

    # Pressure
    "ATM": {"PA": lambda x: x * 101325, "KPA": lambda x: x * 101.325, "BAR": lambda x: x * 1.01325, "MB": lambda x: x * 1013.25, "INHG": lambda x: x * 29.92126},
    "INHG": {"MMHG": lambda x: x * 25.4, "PA": lambda x: x * 3386.38816, "ATM": lambda x: x / 29.92126, "MB": lambda x: x * 33.863886666667},  # Add inHg to mb conversion
    "PSI": {"PA": lambda x: x * 6894.76, "KPA": lambda x: x * 6.89476},
    "MB": {"INHG": lambda x: x / 33.863886666667},  # Add mb to inHg conversion

    # Energy/Work
    "J": {"CAL": lambda x: x * 0.239006, "KCAL": lambda x: x * 0.000239006, "BTU": lambda x: x * 0.000947817},
    "EV": {"J": lambda x: x * 1.60218e-19},

    # Power
    "W": {"HP": lambda x: x * 0.00134102},

    # Angle
    "DEG": {"RAD": lambda x: math.radians(x)},
    "RAD": {"DEG": lambda x: math.degrees(x)},

    # Trigonometry
    "SIN_DEG": {"": lambda x: math.sin(math.radians(x))},
    "COS_DEG": {"": lambda x: math.cos(math.radians(x))},
    "TAN_DEG": {"": lambda x: math.tan(math.radians(x))},
    # Add more trigonometric functions (asin, acos, atan, etc.) similarly

    # Calculus (Basic example for derivative of a polynomial)
    "DERIVATIVE_AT": {
        "": lambda coeffs, point:
            sum(c * i * point**(i-1) for i, c in enumerate(coeffs) if i > 0)
    }
}

@bot.command(name="convert")
async def convert(ctx, *args):
    """
    Converts a value from one unit to another using predefined conversion factors.

    Usage:
    - For direct conversions: !convert <value> <from_unit> <to_unit>
    - For HP to torque: !convert 1 <horsepower> <rpm>
    - For torque to HP: !convert 2 <torque> <rpm>
    """

    if len(args) == 3:  # Direct conversion
        value, from_unit, to_unit = args
        value = float(value)
        from_unit = from_unit.upper()
        to_unit = to_unit.upper()

        if from_unit in conversion_factors and to_unit in conversion_factors[from_unit]:
            try:
                converted_value = conversion_factors[from_unit][to_unit](value)
                embed = discord.Embed(title="Conversion Result", color=discord.Color.orange())
                embed.add_field(name="Original Value", value=f"{value} {from_unit}", inline=True)
                embed.add_field(name="Converted Value", value=f"{converted_value:.4f} {to_unit}", inline=True)
                await ctx.send(embed=embed)
            except (ValueError, ZeroDivisionError) as e:
                await ctx.send(f"Error during conversion: {e}")
        else:
            await ctx.send("Unsupported conversion or units. Please check your input.")

    elif len(args) == 3 and args[0] == '1':  # HP to torque
        try:
            hp, rpm = float(args[1]), float(args[2])
        except ValueError:
            await ctx.send("Invalid input for HP to Torque conversion. Please use: $convert 1 <horsepower> <rpm>")
            return  # Stop further execution if input is invalid

        torque = conversion_factors["HP_TO_TORQUE"](hp, rpm)
        embed = discord.Embed(title="Conversion Result", color=discord.Color.orange())
        embed.add_field(name="Horsepower", value=f"{hp} HP", inline=True)
        embed.add_field(name="RPM", value=f"{rpm} RPM", inline=True)
        embed.add_field(name="Torque", value=f"{torque:.4f} lb-ft", inline=False)  # Specify lb-ft
        await ctx.send(embed=embed)

    elif len(args) == 3 and args[0] == '2':  # Torque to HP
        try:
            torque, rpm = float(args[1]), float(args[2])
        except ValueError:
            await ctx.send("Invalid input for Torque to HP conversion. Please use: $convert 2 <torque> <rpm>")
            return

        hp = conversion_factors["TORQUE_TO_HP"](torque, rpm)
        embed = discord.Embed(title="Conversion Result", color=discord.Color.orange())
        embed.add_field(name="Torque", value=f"{torque} lb-ft", inline=True)
        embed.add_field(name="RPM", value=f"{rpm} RPM", inline=True)
        embed.add_field(name="Horsepower", value=f"{hp:.4f} HP", inline=False)
        await ctx.send(embed=embed)

    else:
        await ctx.send("Invalid conversion format. Please check the usage.")
    
convert.help = """
    **$convert** - Converts values between different units.

    **Usage:**
    $convert <value> <from_unit> <to_unit>

    **Examples:**
    * $convert 100 C F  (Converts 100 degrees Celsius to Fahrenheit)
    * $convert 1 KM MI  (Converts 1 kilometer to miles)
    * $convert 10 KT MPH (Converts 10 knots to miles per hour)
    * $convert 30 DEG RAD (Converts 30 degrees to radians)
    * $convert 60 COS_DEG  (Calculates the cosine of 60 degrees)
    * $convert Derivatives
    * $convert 1 200 5000 (Converts 200 horsepower at 5000 RPM to torque)
    * $convert 2 300 3500 (Converts 300 lb-ft of torque at 3500 RPM to horsepower)

    **Supported Units:**
    * Temperature: C (Celsius), F (Fahrenheit), K (Kelvin)
    * Length/Distance: M (meters), KM (kilometers), FT (feet), MI (miles), NM (nautical miles), AU (astronomical units), LY (light-years), PC (parsecs)
    * Area: ACRE, MI^2, KM^2, HA (hectares)
    * Volume: GAL_US (US gallons), GAL_IMP (imperial gallons), L (liters), FT^3 (cubic feet), M^3 (cubic meters), BBL (barrels of oil)
    * Mass/Weight: LB (pounds), KG (kilograms), OZ (ounces), G (grams), TON_US (US tons), TONNE (metric tons)
    * Force: N (newtons), LBF (pound-force)
    * Pressure: ATM (atmospheres), PA (pascals), KPA (kilopascals), BAR, MB (millibars), INHG (inches of mercury), MMHG (millimeters of mercury), PSI (pounds per square inch)
    * Energy/Work: J (joules), CAL (calories), KCAL (kilocalories), BTU (British thermal units), EV (electronvolts)
    * Power: W (watts), HP (horsepower)
    * Angle: DEG (degrees), RAD (radians)
    * Trigonometry: SIN_DEG (sine of angle in degrees), COS_DEG (cosine of angle in degrees), TAN_DEG (tangent of angle in degrees)
    * Calculus: Derivative
    """
if __name__ == "__main__":
    bot.run(token=config.DISCORD_TOKEN)
