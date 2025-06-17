import os
import sys
import time
import asyncio
import subprocess
import random
import discord
import config
import io
from discord.ext import commands
import requests
import urllib3
import pytz
import shapefile
import zipfile
from bs4 import BeautifulSoup
from io import BytesIO
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
from geopy.geocoders import Nominatim
import astropy.coordinates as coord
from astropy.time import Time
from timezonefinder import TimezoneFinder
import ephem
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import pandas as pd
import geocoder
import json
import psutil
import signal
import math
from siphon.simplewebservice.wyoming import WyomingUpperAir
import metpy
from metpy.units import units
import matplotlib.gridspec as gridspec
from metpy.plots import add_metpy_logo, SkewT, Hodograph
import xarray as xr
# Removed duplicate openmeteo_py imports
import aiohttp
from metpy.calc import parcel_profile, mixed_layer_cape_cin
import certifi
from metpy.plots import ColdFront, WarmFront, OccludedFront, StationaryFront, StationPlot
from shapely.geometry import Point, LineString
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
import metpy.calc as mpcalc
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from metpy.calc import dewpoint_from_relative_humidity
from weather_maps import wind300, wind500, vort500, rh700, wind850, mAdv850, tAdv850, dew850, surfaceTemp
# Ensure these are correctly imported if they are separate files in the same directory
try:
    from utils import parse_date # Assuming utils.py is in the same directory
    from weather_calculations import calc_mslp # Assuming weather_calculations.py is in the same directory
    from RTMA_Graphics import ( # Assuming RTMA_Graphics.py is in the same directory
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
except ImportError as e:
    logging.error(f"Failed to import local modules: {e}. Ensure utils.py, weather_calculations.py, and RTMA_Graphics.py are in the same directory as main_bot_script.py")

import matplotlib as mpl
import re
from fuzzywuzzy import fuzz
import metpy.interpolate as mpinterpolate
from matplotlib.colors import LinearSegmentedColormap
from skewt import skewt # Assuming skewt.py is a local file
from siphon.simplewebservice.iastate import IAStateUpperAir
from datetime import datetime, timedelta, timezone
from metpy.plots import add_timestamp
from scipy.signal import find_peaks
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
# from pyfonts import load_font # Commented out as pyfonts might not be standard
import matplotlib.dates as mdates
from tropycal import tracks, rain, recon
import tropycal.realtime as tropycal_realtime  # Renamed to avoid conflict
from tropycal.rain import RainDataset
from uuid import uuid4
from matplotlib import cm
from matplotlib.colors import Normalize, BoundaryNorm
import cartopy.io.shapereader as shpreader
# Ensure these are correctly imported if they are separate files in the same directory
try:
    from wrd_wx import wrd_wx # Assuming wrd_wx.py is in the same directory
    '''from eu_weather_maps import eu_wind300, eu_vort500, eu_rh700, eu_wind850, eu_surfaceTemp, eu_mAdv850, eu_tAdv850 # Assuming eu_weather_maps.py is in the same directory
    from au_weather_maps import au_wind300, au_vort500, au_rh700, au_wind850, au_mAdv850, au_tAdv850, au_surfaceTemp # Assuming au_weather_maps.py is in the same directory'''
    from hurricane import hurricane # Assuming hurricane.py is in the same directory
    from georgia_temp import georgia_temp # Assuming georgia_temp.py is in the same directory
except ImportError as e:
    logging.error(f"Failed to import regional weather map modules: {e}. Ensure these files are correctly placed.")




OPENWEATHERMAP_API_KEY = 'efd4f5ec6d2b16958a946b2ceb0419a6'

# Set up bot with intents and command prefix
intents = discord.Intents.default()
intents.message_content = True  # Required for prefix commands
bot = commands.Bot(command_prefix='$', intents=intents)



# Set up logging for the main bot file
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Initialize the Discord bot
intents = discord.Intents.default()
intents.message_content = True # Required for reading message content for commands
bot = commands.Bot(command_prefix='$', intents=intents)


# Semaphore to limit concurrent commands
command_semaphore = asyncio.Semaphore(2)  # Allow 2 concurrent commands

@bot.event
async def on_ready():
    """Event handler for when the bot successfully connects to Discord."""
    logging.info(f'{bot.user} has connected to Discord!')
    print(f'{bot.user} has connected to Discord!')


# --- External Commands ---
# Add external commands directly after bot setup
bot.add_command(hurricane)
bot.add_command(georgia_temp)
bot.add_command(wrd_wx)
#bot.add_command(eu_wind300)
#bot.add_command(eu_vort500)
#bot.add_command(eu_rh700)
#bot.add_command(eu_wind850)
#bot.add_command(eu_tAdv850)
#bot.add_command(eu_mAdv850)
#bot.add_command(eu_surfaceTemp)
#bot.add_command(au_wind300)
#bot.add_command(au_vort500)
#bot.add_command(au_rh700)
#bot.add_command(au_wind850)
#bot.add_command(au_mAdv850)
#bot.add_command (au_tAdv850)
#bot.add_command(au_surfaceTemp)

# Register commands
bot.add_command(wind300)
bot.add_command(wind500)
bot.add_command(vort500)
bot.add_command(rh700)
bot.add_command(wind850)
bot.add_command(mAdv850)
bot.add_command(tAdv850)
bot.add_command(dew850)
bot.add_command(surfaceTemp)

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
    await bot.process_commands(message)

# --- Restart Command ---
@bot.command()
async def restart(ctx):
    try:
        await ctx.send("Restarting...")
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            try:
                os.close(handler.fd)
            except Exception as e:
                print(f"Error closing handler: {e}")
        time.sleep(2)
        python = sys.executable
        os.execl(python, python, *sys.argv)
    except Exception as e:
        await ctx.send(f"Error during restart: {e}")

# --- METAR Command ---
def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")
        raw_metars = [entry['rawOb'] for entry in json_data]
        return raw_metars
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

@bot.command()
async def metar(ctx, airport_code: str, hoursback: int = 0):
    airport_code = airport_code.upper()
    try:
        raw_metars = get_metar(airport_code, hoursback)
        embed = discord.Embed(title=f"METAR for {airport_code}", color=0x01c0db)
        if hoursback > 0:
            embed.title = f"METARs for {airport_code} (Last {hoursback} Hours)"
            for i, raw_metar in enumerate(raw_metars):
                embed.add_field(name=f"Observation {i+1}", value=f"`\n{raw_metar}\n`", inline=False)
        else:
            embed.description = f"`\n{raw_metars[0]}\n`"
        await ctx.send(embed=embed)
        logging.info(f"User {ctx.author} requested METAR for {airport_code} (hoursback={hoursback})")
    except Exception as e:
        await ctx.send(f"Error fetching METAR: {e}")


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

# --- METAR Utility Functions (as provided by you) ---

def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)

        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

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

    cloud_pattern = re.compile(r'((FEW|SCT|BKN|OVC)(\d{3}))|(VV(\d{3}))')
    cloud_matches = re.findall(cloud_pattern, metar)

    for match in cloud_matches:
        if match[1]:
            cover = match[1]
            altitude_hundreds = int(match[2])
            altitude_ft = altitude_hundreds * 100

            if altitude_ft <= 6500:
                cloud_levels["low"].append((cover, altitude_ft))
            elif 6500 < altitude_ft <= 20000:
                cloud_levels["mid"].append((cover, altitude_ft))
            else:
                cloud_levels["high"].append((cover, altitude_ft))

        if match[4]:
            vv_hundreds = int(match[4])
            cloud_levels["vertical_visibility"] = vv_hundreds * 100

    return cloud_levels

# Function to convert pressure from altimeter setting (Axxxx) to inches of mercury
def convert_pressure(altimeter_str):
    return float(altimeter_str[1:]) / 100

# Function to extract wind information from a METAR string
def extract_wind_info(metar):
    wind_direction = -999
    wind_speed = -999
    wind_gusts = np.nan

    wind_match = re.search(r'(\d{3})(\d{2})(G\d{2})?KT', metar)
    if wind_match:
        wind_direction = int(wind_match.group(1))
        wind_speed = int(wind_match.group(2))
        if wind_match.group(3):
            wind_gusts = int(wind_match.group(3)[1:])

    wind_speed = max(wind_speed, 0)
    wind_gusts = max(wind_gusts, 0) if not np.isnan(wind_gusts) else np.nan

    return wind_direction, wind_speed, wind_gusts

# --- GLOBAL VARIABLES for Precipitation Parsing ---
PRECIP_CODES = [
    'DZ', 'RA', 'SN', 'SG', 'IC', 'PL', 'GR', 'GS', 'UP',
    'SH', 'TS',
    'FZ'
]
PRECIP_REGEX = r'(?<!VC)([+-]?(?:' + '|'.join(PRECIP_CODES) + r'))'


# --- UPDATED process_metar_data Function ---
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
        "vertical_visibility": [],
        "present_weather_codes": [], # New: Store all present weather codes as a list
        "hourly_precipitation_in": [] # New: Store parsed hourly precipitation amount
    }

    now_utc = datetime.now(timezone.utc)

    print("Raw METAR data received:")
    for metar in metar_list:
        print(metar)

    for metar in metar_list:
        parts = metar.split()

        # Extract observation time
        try:
            observation_day = int(parts[1][0:2])
            observation_hour = int(parts[1][2:4])
            observation_minute = int(parts[1][4:6])

            temp_dt = now_utc.replace(day=observation_day,
                                     hour=observation_hour,
                                     minute=observation_minute,
                                     second=0, microsecond=0)

            if temp_dt > now_utc + timedelta(days=1):
                if temp_dt.month == 1:
                    temp_dt = temp_dt.replace(year=temp_dt.year - 1, month=12)
                else:
                    temp_dt = temp_dt.replace(month=temp_dt.month - 1)
            data["time"].append(temp_dt)

        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse time from METAR '{metar}': {e}")
            data["time"].append(np.nan)
            # IMPORTANT: Append NaNs for other data points to maintain list length
            # if time parsing fails to prevent "All arrays must be of the same length"
            data["temperature"].append(np.nan)
            data["dewpoint"].append(np.nan)
            data["wind_direction"].append(np.nan)
            data["wind_speed"].append(np.nan)
            data["wind_gusts"].append(np.nan)
            data["pressure"].append(np.nan)
            data["low_clouds"].append(np.nan)
            data["mid_clouds"].append(np.nan)
            data["high_clouds"].append(np.nan)
            data["vertical_visibility"].append(np.nan)
            data["present_weather_codes"].append([]) # Append empty list if time fails
            data["hourly_precipitation_in"].append(np.nan)
            continue # Skip to next METAR if time parsing fails completely

        # Extract temperature and dewpoint
        temp_dewpoint_match = re.search(r'([M]?\d{1,2})/([M]?\d{1,2})', metar)
        if temp_dewpoint_match:
            temp_str, dewpoint_str = temp_dewpoint_match.groups()
            temp = int(temp_str.replace('M', '-')) if temp_str else np.nan
            dewpoint = int(dewpoint_str.replace('M', '-')) if dewpoint_str else np.nan
            if not np.isnan(temp) and not np.isnan(dewpoint) and dewpoint > temp:
                dewpoint = temp
            data["temperature"].append(temp)
            data["dewpoint"].append(dewpoint)
        else:
            data["temperature"].append(np.nan)
            data["dewpoint"].append(np.nan)

        # Wind information
        direction, speed, gusts = extract_wind_info(metar)
        data["wind_direction"].append(direction if direction != -999 else np.nan)
        data["wind_speed"].append(speed if speed != -999 else np.nan)
        data["wind_gusts"].append(gusts if gusts != -999 else np.nan)

        # Pressure extraction
        pressure_match = re.search(r'A(\d{4})', metar)
        pressure_inhg = float(pressure_match.group(1)) / 100 if pressure_match else np.nan
        pressure_hpa = pressure_inhg * 33.8639 if not np.isnan(pressure_inhg) else np.nan
        data["pressure"].append(pressure_hpa)

        # Cloud cover and vertical visibility extraction
        cloud_info = extract_cloud_info(metar)
        low_clouds = cloud_info["low"][0][1] if cloud_info["low"] else np.nan
        mid_clouds = cloud_info["mid"][0][1] if cloud_info["mid"] else np.nan
        high_clouds = cloud_info["high"][0][1] if cloud_info["high"] else np.nan
        vertical_visibility = cloud_info["vertical_visibility"] if cloud_info["vertical_visibility"] else np.nan

        data["low_clouds"].append(low_clouds)
        data["mid_clouds"].append(mid_clouds)
        data["high_clouds"].append(high_clouds)
        data["vertical_visibility"].append(vertical_visibility)

        # --- NEW: Precipitation Extraction ---
        # 1. Present Weather (e.g., RA, SN, +TSRA)
        present_weather_matches = re.findall(PRECIP_REGEX, metar)
        valid_present_weather = [
            pw for pw in present_weather_matches if pw and any(code in pw for code in PRECIP_CODES)
        ]
        data["present_weather_codes"].append(valid_present_weather)

        # 2. Hourly Precipitation Amount (P####)
        hourly_precip_match = re.search(r'P(\d{4})', metar)
        if hourly_precip_match:
            hourly_precip = float(hourly_precip_match.group(1)) / 100
            data["hourly_precipitation_in"].append(hourly_precip)
        else:
            data["hourly_precipitation_in"].append(np.nan)
        # --- END NEW: Precipitation Extraction ---


    # Create DataFrame from parsed data
    df = pd.DataFrame(data)

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)

    df.sort_values(by='time', inplace=True)

    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['dewpoint'] = pd.to_numeric(df['dewpoint'], errors='coerce')
    df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')

    window_size = 5
    df['temperature_filtered'] = df['temperature'].rolling(window=window_size, center=True).median()
    df['dewpoint_filtered'] = df['dewpoint'].rolling(window=window_size, center=True).median()

    temp_spikes = (df['temperature'] - df['temperature_filtered']).abs() > 5
    dewpoint_spikes = (df['dewpoint'] - df['dewpoint_filtered']).abs() > 5

    df.loc[temp_spikes, 'temperature'] = df['temperature_filtered']
    df.loc[dewpoint_spikes, 'dewpoint'] = df['dewpoint_filtered']

    smoothing_window = 3
    df['temperature_smoothed'] = df['temperature'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df['dewpoint_smoothed'] = df['dewpoint'].rolling(window=smoothing_window, center=True, min_periods=1).mean()

    df['temperature_smoothed'] = df['temperature_smoothed'].interpolate(method='linear')
    df['dewpoint_smoothed'] = df['dewpoint_smoothed'].interpolate(method='linear')

    df.dropna(subset=['temperature_smoothed', 'dewpoint_smoothed'], inplace=True)

    df['relative_humidity'] = mpcalc.relative_humidity_from_dewpoint(
        df['temperature_smoothed'].values * units.degC,
        df['dewpoint_smoothed'].values * units.degC
    ) * 100

    print("Relative Humidity (%):", df['relative_humidity'].values)

    wet_bulb_values = []
    for index, row in df.iterrows():
        temp = row['temperature_smoothed']
        dewpoint = row['dewpoint_smoothed']
        pressure = row['pressure']

        if not np.isnan(temp) and not np.isnan(dewpoint) and not np.isnan(pressure):
            try:
                wet_bulb_temp = mpcalc.wet_bulb_temperature(
                    pressure * units.hPa,
                    temp * units.degC,
                    dewpoint * units.degC
                ).to('degF').m
                wet_bulb_values.append(wet_bulb_temp)
            except Exception as e:
                print(f"Error calculating wet bulb temperature at index {index}: {e}")
                wet_bulb_values.append(np.nan)
        else:
            wet_bulb_values.append(np.nan)

    df['wet_bulb'] = wet_bulb_values

    print("DataFrame summary before plotting:")
    print(df[['time', 'temperature', 'dewpoint', 'pressure', 'relative_humidity', 'wet_bulb', 'present_weather_codes', 'hourly_precipitation_in']])

    return df

# --- Environmental Calculations (as provided by you) ---

def to_heat_index(tempF, dewF):
    tempF = np.array(tempF)
    dewF = np.array(dewF)

    # Convert to Celsius for mpcalc
    tempC = (tempF - 32) * 5 / 9
    dewC = (dewF - 32) * 5 / 9

    # Calculate RH properly using mpcalc
    rh = mpcalc.relative_humidity_from_dewpoint(
        tempC * units.degC,
        dewC * units.degC
    ) * 100
    rh = np.clip(rh.magnitude, 0, 100)  # Clamp RH to 0-100%

    # Heat index coefficients
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 * (10 ** -3)
    c6 = -5.481717 * (10 ** -2)
    c7 = 1.22874 * (10 ** -3)
    c8 = 8.5282 * (10 ** -4)
    c9 = -1.99 * (10 ** -6)

    heat_index = (c1 + (c2 * tempF) + (c3 * rh) + (c4 * tempF * rh) +
                  (c5 * tempF ** 2) + (c6 * rh ** 2) + (c7 * tempF ** 2 * rh) +
                  (c8 * tempF * rh ** 2) + (c9 * tempF ** 2 * rh ** 2))

    # Only calculate heat index for tempF >= 80°F, otherwise return NaN
    heat_index = np.where(tempF >= 80, heat_index, np.nan)
    return heat_index

def to_wind_chill(tempF, wind_speed):
    wind_speed_mph = wind_speed * 1.15078
    wind_chill = np.where((tempF <= 50) & (wind_speed > 5),
                          35.74 + 0.6215 * tempF - 35.75 * wind_speed_mph**0.16 + 0.4275 * tempF * wind_speed_mph**0.16,
                          np.nan)
    return wind_chill

# --- Discord Bot Command ---

@bot.command()
async def meteogram(ctx, icao: str, hoursback: int):
    try:
        metar_list = get_metar(icao, hoursback)
        df = process_metar_data(metar_list)

        utc_time = datetime.now(pytz.utc)

        if df.empty:
            await ctx.send("No valid METAR data available for plotting.")
            return

        df['tempF'] = (df['temperature_smoothed'] * 9 / 5) + 32  # Use smoothed temps for plots
        df['dewF'] = (df['dewpoint_smoothed'] * 9 / 5) + 32  # Use smoothed dewpoints for plots

        # Calculate heat index, wind chill, etc.
        df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)
        df.loc[df['tempF'] < 80, 'heat_index'] = np.nan

        df['wind_chill'] = to_wind_chill(df['tempF'].values, df['wind_speed'].values)
        df.loc[(df['wind_speed'] <= 5) | (df['tempF'] > 50), 'wind_chill'] = np.nan

        # Calculate statistics for text box
        if not df.empty:
            max_temp = df['tempF'].max()
            min_temp = df['tempF'].min()
            avg_wind_speed = df['wind_speed'].mean()
            if not df['wind_direction'].dropna().empty:
                wind_dirs_rad = np.deg2rad(df['wind_direction'].dropna())
                mean_sin = np.mean(np.sin(wind_dirs_rad))
                mean_cos = np.mean(np.cos(wind_dirs_rad))
                avg_wind_dir = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
            else:
                avg_wind_dir = np.nan
            avg_rh = df['relative_humidity'].mean()
            max_gust = df['wind_gusts'].max() if 'wind_gusts' in df.columns else np.nan

            # Precipitation totals
            precip_data = df.dropna(subset=['hourly_precipitation_in'])
            total_rain = 0
            total_snow = 0
            total_freezing_rain = 0
            total_sleet = 0
            total_precip = 0
            for index, row in precip_data.iterrows():
                precip_amount = row['hourly_precipitation_in']
                weather_codes = row['present_weather_codes']
                if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA', 'TSRA']):
                    total_rain += precip_amount
                elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']):
                    total_snow += precip_amount
                elif any(code in weather_codes for code in ['FZRA', 'FZDZ']):
                    total_freezing_rain += precip_amount
                elif any(code in weather_codes for code in ['PL', 'GS']):
                    total_sleet += precip_amount
                total_precip += precip_amount
        else:
            max_temp = min_temp = avg_wind_speed = avg_wind_dir = avg_rh = max_gust = total_rain = total_snow = total_freezing_rain = total_sleet = total_precip = np.nan

        # Create subplots
        fig, axs = plt.subplots(6, 1, figsize=(20, 20), sharex=True)
        fig.patch.set_facecolor('lightsteelblue')

        for ax in axs:
            ax.grid(True)

        axs[0].set_title(f'Meteogram for {icao.upper()} - Last {hoursback} hours (Generated at: {utc_time.strftime("%Y-%m-%d %H:%M UTC")})', weight='bold', size='16')

        # 0. Temperature, Dewpoint, Wet Bulb, Heat Index, Wind Chill
        if not df[['tempF', 'dewF']].isnull().all().any():
            plt.sca(axs[0])
            plt.plot(df['time'], df['tempF'], label='Temperature (°F)', linewidth=3, color='tab:red')
            plt.plot(df['time'], df['dewF'], label='Dewpoint (°F)', linewidth=3, color='tab:green')
            plt.plot(df['time'], df['wet_bulb'], label='Wet Bulb (°F)', linewidth=3, linestyle='dotted', color='tab:blue')
            plt.plot(df['time'], df['heat_index'], label='Heat Index (°F)', linestyle='--', color='tab:orange')
            plt.plot(df['time'], df['wind_chill'], label='Wind Chill (°F)', linestyle='--', color='tab:purple')
            axs[0].set_ylabel('Temperature (°F)')
            axs[0].legend(loc='upper left', fontsize=10, frameon=True, title='Temperature / Dewpoint')
        else:
            axs[0].axhline(0, color='black')
            axs[0].set_ylabel('Temperature (°F)')

        # 1. Wind Speed and Gusts
        valid_wind_data = df.dropna(subset=['wind_direction', 'wind_speed'], how='any')
        if not valid_wind_data.empty:
            vmin = valid_wind_data['wind_speed'].min()
            vmax = valid_wind_data['wind_speed'].max()
            scatter = axs[1].scatter(valid_wind_data['time'], valid_wind_data['wind_direction'],
                                     c=valid_wind_data['wind_speed'], cmap='brg', label='Wind Speed (knots)',
                                     vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10)
            gust_data = valid_wind_data.dropna(subset=['wind_gusts'])
            if not gust_data.empty:
                axs[1].scatter(gust_data['time'], gust_data['wind_direction'],
                               c=gust_data['wind_gusts'], cmap='Greys', marker='x', s=100,
                               label='Wind Gusts (knots)', vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10)
            cbar = plt.colorbar(scatter, ax=axs[1], pad=0.002)
            cbar.set_label('Wind Speed (knots)')
            axs[1].set_yticks([0, 90, 180, 270, 360])
            axs[1].set_yticklabels(['N', 'E', 'S', 'W', 'N'])
            axs[1].set_ylim(0, 360)
            axs[1].set_ylabel('Wind Direction (°)')
            axs[1].legend(loc='upper left', fontsize=10, frameon=True, title='Wind')
        else:
            axs[1].axhline(0, color='black')
            axs[1].set_ylabel('Wind Direction (°)')

        # 2. Pressure
        if not df['pressure'].isnull().all():
            axs[2].plot(df['time'], df['pressure'], label='Pressure (hPa)', linewidth=3, color='tab:purple')
            axs[2].set_ylabel('Pressure (hPa)')
            axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[2].legend(loc='upper left', fontsize=10, frameon=True, title='Pressure')
        else:
            axs[2].axhline(0, color='black')
            axs[2].set_ylabel('Pressure (hPa)')

        # 3. Cloud Cover and Vertical Visibility
        axs[3].scatter(df['time'], df['low_clouds'], color='green', label='Low Clouds', marker='s')
        axs[3].scatter(df['time'], df['mid_clouds'], color='blue', label='Mid Clouds', marker='o')
        axs[3].scatter(df['time'], df['high_clouds'], color='red', label='High Clouds', marker='^')
        axs[3].scatter(df['time'], df['vertical_visibility'], color='yellow', label='Vertical Visibility', marker='v')
        axs[3].set_ylabel('Cloud Cover / Vertical Visibility (ft)')
        axs[3].legend(loc='upper left', fontsize=10, frameon=True, title='Cloud Cover')
        axs[3].set_ylim(0)

        # 4. Precipitation (Bar chart with color-coded bars)
        precip_data = df.dropna(subset=['hourly_precipitation_in'])
        if not precip_data.empty:
            bar_width = timedelta(hours=0.8)
            for index, row in precip_data.iterrows():
                precip_amount = row['hourly_precipitation_in']
                weather_codes = row['present_weather_codes']
                timestamp = row['time']
                color = 'gray'
                if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA', 'TSRA']):
                    color = 'green'
                elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']):
                    color = 'blue'
                elif any(code in weather_codes for code in ['FZRA', 'FZDZ']):
                    color = 'purple'
                elif any(code in weather_codes for code in ['PL', 'GS']):
                    color = 'pink'
                axs[4].bar(timestamp, precip_amount, width=bar_width, color=color, zorder=2)
            axs[4].set_ylabel('Precipitation (in)')
            axs[4].set_ylim(bottom=0)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Rain'),
                Patch(facecolor='blue', label='Snow'),
                Patch(facecolor='purple', label='Freezing Rain'),
                Patch(facecolor='pink', label='Sleet')
            ]
            axs[4].legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, title='Precipitation Types')
        else:
            axs[4].axhline(0, color='black')
            axs[4].set_ylabel('Precipitation (in)')

        # Add icons for present weather
        icon_y_offset = 0.07
        icon_size = 0.07
        for index, row in df.iterrows():
            timestamp = row['time']
            weather_codes = row['present_weather_codes']
            if weather_codes and pd.notna(timestamp):
                icon_path = None
                if 'RA' in weather_codes or '+RA' in weather_codes or '-RA' in weather_codes:
                    icon_path = 'meteogram_rain_icon.png'
                elif 'SN' in weather_codes or '+SN' in weather_codes or '-SN' in weather_codes:
                    icon_path = 'meteogram_snow_icon.png'
                elif 'TS' in weather_codes or 'TSRA' in weather_codes:
                    icon_path = 'meteogram_thunderstorm_icon.png'
                elif 'DZ' in weather_codes:
                    icon_path = 'meteogram_drizzle_icon.png'
                elif 'FZRA' in weather_codes or 'FZDZ' in weather_codes:
                    icon_path = 'meteogram_freezing_rain_icon.png'
                if icon_path:
                    try:
                        img = plt.imread(icon_path)
                        im = OffsetImage(img, zoom=icon_size)
                        ab = AnnotationBbox(im, (timestamp, axs[4].get_ylim()[0] + icon_y_offset * (axs[4].get_ylim()[1] - axs[4].get_ylim()[0])),
                                            xycoords='data', frameon=False, box_alignment=(0.5, 0))
                        axs[4].add_artist(ab)
                    except FileNotFoundError:
                        print(f"Icon file not found: {icon_path}. Skipping icon for {weather_codes} at {timestamp}.")

        # 5. Relative Humidity
        if not df['relative_humidity'].isnull().all():
            rh_array = df['relative_humidity'].values
            rh_band = 5
            rh_lower = np.maximum(0, rh_array - rh_band)
            rh_upper = np.minimum(100, rh_array + rh_band)
            axs[5].fill_between(df['time'], rh_lower, rh_upper, color='cornflowerblue', alpha=0.3, label='RH Range')
            axs[5].plot(df['time'], rh_array, color='darkblue', label='Relative Humidity (%)', linewidth=2)
            axs[5].set_ylabel('Relative Humidity (%)')
            axs[5].legend(loc='upper left', fontsize=10, frameon=True, title='Relative Humidity')
            axs[5].set_ylim(0, 100)
        else:
            axs[5].axhline(0, color='black')
            axs[5].set_ylabel('Relative Humidity (%)')

        # Add text box below relative humidity subplot (axs[5])
        if not df.empty:
            precip_text = (
                f"Total Precipitation:\n"
                f"  Rain: {total_rain:.2f} in\n"
                f"  Snow: {total_snow:.2f} in\n"
                f"  Freezing Rain: {total_freezing_rain:.2f} in\n"
                f"  Sleet: {total_sleet:.2f} in\n"
                f"  Sum: {total_precip:.2f} in\n\n"
                f"Temperature:\n"
                f"  Max: {max_temp:.1f} °F\n"
                f"  Min: {min_temp:.1f} °F\n\n"
                f"Average Wind:\n"
                f"  Speed: {avg_wind_speed:.1f} knots\n"
                f"  Direction: {avg_wind_dir:.0f}°\n\n"
                f"Average Relative Humidity: {avg_rh:.1f}%\n"
            )
            if not np.isnan(max_gust):
                precip_text += f"Max Wind Gust: {max_gust:.1f} knots\n"
            axs[5].text(
                0.5, -0.35, precip_text, transform=axs[5].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
            )

        # Control x-axis ticks and labels
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))

        for ax in axs[:-1]:
            ax.xaxis.set_tick_params(labelbottom=True)

        plt.setp(axs[-1].get_xticklabels(), rotation=60, fontsize=8)
        plt.setp([ax.get_xticklabels() for ax in axs[:-1]], rotation=60, fontsize=8)

        # Refine grid and background
        for ax in axs:
            ax.grid(True, zorder=0)
            ymin, ymax = ax.get_ylim()
            y_ticks = ax.get_yticks()
            if len(y_ticks) > 1:
                y_ticks_filtered = [y for y in y_ticks if ymin <= y <= ymax]
                if len(y_ticks_filtered) > 1:
                    for i in range(len(y_ticks_filtered) - 1):
                        if i % 2 == 0:
                            ax.axhspan(y_ticks_filtered[i], y_ticks_filtered[i+1], color='limegreen', alpha=0.3, zorder=1)

        # Adjust subplot spacing
        fig.subplots_adjust(hspace=0.4, left=0.05, right=0.97, top=0.95, bottom=0.15)

        '''# METOC Logo
        logo_img = plt.imread('/media/evanl/BACKUP/bot/boxlogo2.png')
        imagebox = OffsetImage(logo_img, zoom=0.125)
        ab = AnnotationBbox(imagebox, (0.98, 0.98), xycoords='figure fraction', frameon=False, box_alignment=(1, 1))
        fig.add_artist(ab)

        # UGA Logo
        usmc_img = plt.imread('/media/evanl/BACKUP/bot/metoc.png')
        imagebox = OffsetImage(usmc_img, zoom=0.075)
        abx = AnnotationBbox(imagebox, (0.995, 0.90), xycoords='figure fraction', frameon=False, box_alignment=(1, 1))
        fig.add_artist(abx)'''

        # Save and send
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150)
        img_buf.seek(0)
        await ctx.send(file=discord.File(img_buf, filename=f'meteogram_{icao}.png'))
        plt.close(fig)

    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
##########################################################################################################


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


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
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
    """
    Displays tropical cyclone rainfall data for a historical storm using
    Tropycal's rain module (Stage IV or IMERG data).
    """

    try:
        # 1) Load the HURDAT2 dataset for the desired basin
        basin = tracks.TrackDataset(basin='north_atlantic')

        # 2) Get the storm from the dataset
        storm = basin.get_storm((storm_name, year))

        if storm is None:
            await ctx.send(f"Storm '{storm_name}' from {year} not found in the dataset.")
            return

        # 3) Prepare an Embed for Discord
        embed = discord.Embed(
            title=f"{storm_name.title()} ({year}) - Rainfall Data",
            color=discord.Color.blue()
        )

        # 4) Figure out an approximate date range for the storm from Tropycal's data
        start_date = storm.dict['time'][0].strftime('%Y-%m-%d')
        end_date = storm.dict['time'][-1].strftime('%Y-%m-%d')

        # 5) Try to retrieve rainfall data
        rain_dataset = RainDataset(data_path='wpc')  # Use 'wpc' for Weather Prediction Center data
        try:
            rain_data = rain_dataset.get_storm_rainfall(storm)
        except Exception as e:
            print(f"Error downloading rainfall data: {e}")
            await ctx.send("Rainfall data could not be retrieved. It may be unavailable for this storm.")
            return

        # 6) Create a file path to save the resulting plot
        rainfall_image_path = f'/home/evanl/Documents/{storm_name}_{year}_rainfall.png'

        # 7) Plot the rainfall on a dynamic domain around the storm
        try:
            fig = plt.figure(figsize=(26, 16))  # Create a new figure
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # Add a GeoAxes
            rain_dataset.plot_rain(storm, ax=ax, save_path=rainfall_image_path)
            plt.close(fig)  # Close the figure after saving

        except Exception as e:
            print(f"Error generating the rainfall plot: {e}")
            await ctx.send("An error occurred while generating the rainfall plot.")
            return

        # 8) Send the plot to Discord
        try:
            with open(rainfall_image_path, 'rb') as f:
                file = discord.File(f, filename=f'{storm_name}_{year}_rainfall.png')
                embed.set_image(url=f'attachment://{storm_name}_{year}_rainfall.png')
                await ctx.send(file=file, embed=embed)

        except Exception as e:
            print(f"Error sending the rainfall plot to Discord: {e}")
            await ctx.send("An error occurred while sending the rainfall plot to Discord.")

    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("An error occurred while fetching storm data. Please try again later.")

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
        photo = Image.open('/home/evanl/Documents/boxlogo2.png')

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
        valid_regions = ["conus", "fulldisk", "mesosector", "mesosector2", "tropicalatlantic", "goam", "ne", "indian", "capeverde", "neatl", 'fl', 'pacus', 'wc', 'ak', 'wmesosector', 'wmesosector2']

        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")

        # Product codes for different regions (updated with new regions and product codes, and GeoColor removed)
        product_codes = {
            "conus": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "fulldisk": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 5: "RGB Air Mass"},
            "mesosector": {2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "mesosector2": {2: "Red Visible", 13: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
            "tropicalatlantic": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor", 22: "RGB"},
            "goam": {2: "Red Visible", 14: "Clean Longwave Infrared Window", 9: "Mid-level Water Vapor"},
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
                "goam": {
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
    *   `goam`: Gulf of America (GOES-16)
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

    *   **GOAM & NE:**
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
import ephem
from matplotlib.gridspec import GridSpec


# Define a custom converter for time
class TimeConverter(commands.Converter):
    async def convert(self, ctx, argument):
        try:
            hour, minute = map(int, argument.split(':'))
            return datetime(year=2024, month=1, day=1, hour=hour, minute=minute).time()
        except ValueError:
            raise commands.BadArgument("Invalid time format. Please use 'HH:MM' (e.g., '14:30').")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom time converter
class TimeConverter(commands.Converter):
    async def convert(self, ctx, argument):
        try:
            hour, minute = map(int, argument.split(':'))
            return datetime(year=2024, month=1, day=1, hour=hour, minute=minute).time()
        except ValueError:
            raise commands.BadArgument("Invalid time format. Use 'HH:MM' (e.g., '14:30').")

@bot.command()
async def astro(ctx, location: str = None, time: TimeConverter = None):
    """Provides sunrise, sunset, moon phase, twilight info, and solar system overview for a given location and time."""
    if not location:
        await ctx.send("Please provide an ICAO code or lat/lon (e.g., '$astro kmge' or '$astro 34.05/-118.25')")
        return

    try:
        # Parse location input
        if '/' in location:
            lat, lon = map(float, location.split('/'))
            lat, lon = round(lat, 4), round(lon, 4)
            location_str = f"Lat: {lat}, Lon: {lon}"
        else:
            airports = airportsdata.load('icao')
            airport = airports.get(location.upper())
            if not airport:
                raise ValueError("Airport not found.")
            lat, lon = round(airport['lat'], 4), round(airport['lon'], 4)
            location_str = location.upper()

        # Determine timezone and time
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lng=lon, lat=lat)
        local_time = (datetime.now(pytz.timezone(timezone)).replace(hour=time.hour, minute=time.minute, second=0, microsecond=0)
                      if time else datetime.now(pytz.timezone(timezone)))
        now = local_time.astimezone(pytz.utc)
        now_local = now.astimezone(pytz.timezone(timezone))

        # Set up observer
        obs = ephem.Observer()
        obs.lat, obs.lon = str(lat), str(lon)
        obs.date = ephem.Date(now)

        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor('lightsteelblue')
        gs = GridSpec(2, 1, height_ratios=[2, 1])

        # --- Sun and Moon Plot (Top Subplot) ---
        ax1 = fig.add_subplot(gs[0])
        ax1.patch.set_facecolor('white')
        sun, moon = ephem.Sun(), ephem.Moon()

        # Calculate sunrise, sunset, and twilight times
        sunrise = ephem.localtime(obs.next_rising(sun))
        sunset = ephem.localtime(obs.next_setting(sun))
        if float(lat) < 0:  # Southern Hemisphere adjustment
            sunrise, sunset = sunset, sunrise

        obs.horizon = '-0:34'  # Civil twilight
        civil_twilight_begin = ephem.localtime(obs.previous_rising(sun, use_center=True))
        civil_twilight_end = ephem.localtime(obs.next_setting(sun, use_center=True))
        obs.horizon = '-6'  # Nautical twilight
        nautical_twilight_begin = ephem.localtime(obs.previous_rising(sun, use_center=True))
        nautical_twilight_end = ephem.localtime(obs.next_setting(sun, use_center=True))
        obs.horizon = '-12'  # Astronomical twilight
        astronomical_twilight_begin = ephem.localtime(obs.previous_rising(sun, use_center=True))
        astronomical_twilight_end = ephem.localtime(obs.next_setting(sun, use_center=True))

        # Moon phase
        moon.compute(obs)
        moon_phase = moon.phase
        moon_phase_text = f"Moon Phase: {moon_phase:.1f}% illuminated{' (Waning)' if moon_phase < 50 else ' (Waxing)'}"

        # Generate sun and moon paths
        times = [now + timedelta(minutes=15 * i) for i in range(96)]

        # Sun path with segmenting
        sun_segments = []
        current_segment = []
        current_linestyle = None
        for time in times:
            obs.date = ephem.Date(time)
            sun.compute(obs)
            azimuth = np.degrees(sun.az) % 360
            altitude = np.degrees(sun.alt)
            linestyle = '-' if altitude >= 0 else '--'
            if current_segment and abs(azimuth - current_segment[-1][0]) > 180:
                sun_segments.append((current_segment, current_linestyle))
                current_segment = []
            if current_linestyle and linestyle != current_linestyle:
                sun_segments.append((current_segment, current_linestyle))
                current_segment = []
            current_linestyle = linestyle
            current_segment.append((azimuth, altitude))
        if current_segment:
            sun_segments.append((current_segment, current_linestyle))

        for segment, linestyle in sun_segments:
            if len(segment) < 2:
                continue
            azimuths, altitudes = zip(*segment)
            ax1.plot(azimuths, altitudes, color='orange', lw=3, linestyle=linestyle)

        # Moon path with segmenting
        moon_segments = []
        current_segment = []
        current_linestyle = None
        for time in times:
            obs.date = ephem.Date(time)
            moon.compute(obs)
            azimuth = np.degrees(moon.az) % 360
            altitude = np.degrees(moon.alt)
            linestyle = '-' if altitude >= 0 else '--'
            if current_segment and abs(azimuth - current_segment[-1][0]) > 180:
                moon_segments.append((current_segment, current_linestyle))
                current_segment = []
            if current_linestyle and linestyle != current_linestyle:
                moon_segments.append((current_segment, current_linestyle))
                current_segment = []
            current_linestyle = linestyle
            current_segment.append((azimuth, altitude))
        if current_segment:
            moon_segments.append((current_segment, current_linestyle))

        for segment, linestyle in moon_segments:
            if len(segment) < 2:
                continue
            azimuths, altitudes = zip(*segment)
            ax1.plot(azimuths, altitudes, color='gray', lw=3, linestyle=linestyle)

        # Current positions
        obs.date = ephem.Date(now)
        sun.compute(obs)
        moon.compute(obs)
        current_sun_az, current_sun_alt = np.degrees(sun.az), np.degrees(sun.alt)
        current_moon_az, current_moon_alt = np.degrees(moon.az), np.degrees(moon.alt)

        # Plot twilight and night zones
        ax1.axhspan(-90, -12, color='midnightblue', alpha=0.3, label='Astronomical Twilight')
        ax1.axhspan(-12, -6, color='deepskyblue', alpha=0.3, label='Nautical Twilight')
        ax1.axhspan(-6, -0.34, color='lightskyblue', alpha=0.3, label='Civil Twilight')
        ax1.axhspan(-90, -18, color='black', alpha=0.5, label='Night')
        ax1.scatter(current_sun_az, current_sun_alt, color='yellow', edgecolors='black', s=150, label='Current Sun Position', zorder=3)
        ax1.scatter(current_moon_az, current_moon_alt, color='gray', edgecolors='black', s=150, label='Current Moon Position', zorder=3)
        ax1.axhline(0, color='blue', linestyle='--', lw=1.5, label='Horizon')

        # Legend entries for paths
        ax1.plot([], [], color='orange', lw=3, linestyle='-', label='Sun Path (above horizon)')
        ax1.plot([], [], color='orange', lw=3, linestyle='--', label='Sun Path (below horizon)')
        ax1.plot([], [], color='gray', lw=3, linestyle='-', label='Moon Path (above horizon)')
        ax1.plot([], [], color='gray', lw=3, linestyle='--', label='Moon Path (below horizon)')

        # Calculate solar noon and optimal solar panel angle
        solar_noon = ephem.localtime(obs.next_transit(sun))
        obs.date = ephem.Date(solar_noon)
        sun.compute(obs)
        optimal_angle = np.degrees(float(obs.lat))

        # Text box with astronomical data
        textstr = (f"Location: {location_str}\n"
                   f"Sun Azimuth: {current_sun_az:.1f}°\nMoon Azimuth: {current_moon_az:.1f}°\nMoon Elevation: {current_moon_alt:.1f}°\n"
                   f"{moon_phase_text}\n\nSunrise: {sunrise.strftime('%Y-%m-%d %I:%M %p %Z')}\nSunset: {sunset.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
                   f"Civil Twilight Begin: {civil_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\nCivil Twilight End: {civil_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
                   f"Nautical Twilight Begin: {nautical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\nNautical Twilight End: {nautical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
                   f"Astronomical Twilight Begin: {astronomical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\nAstronomical Twilight End: {astronomical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
                   f"Optimal Solar Panel Angle: ({optimal_angle:.1f}°)")
        plt.figtext(0.05, 0.05, textstr, fontsize=9, weight='bold', va="bottom", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # Customize sun/moon plot
        ax1.set_title(f"Sun/Moon Position: {location_str} - {lat:.4f}, {lon:.4f}\n{now_local.strftime('%b %d %Y %H:%M')} Local Time", fontsize=14, weight='bold')
        ax1.set_xlabel("Azimuth (degrees)", weight='bold')
        ax1.set_ylabel("Altitude (degrees)", weight='bold')
        ax1.set_xlim(0, 360)
        ax1.set_ylim(-90, 90)
        ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax1.set_xticklabels(['N\n0°', 'NE\n45°', 'E\n90°', 'SE\n135°', 'S\n180°', 'SW\n225°', 'W\n270°', 'NW\n315°', 'N\n360°'])
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- Solar System Overview (Bottom Subplot) ---
        ax3 = fig.add_subplot(gs[1], projection='polar')
        ax3.scatter(0, 0, color='yellow', s=100, edgecolor='black', label='Sun', zorder=3)

        # Define planets
        planets = {
            'Mercury': ephem.Mercury(), 'Venus': ephem.Venus(), 'Earth': 'Earth', 'Mars': ephem.Mars(),
            'Jupiter': ephem.Jupiter(), 'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus(), 'Neptune': ephem.Neptune(), 'Pluto': ephem.Pluto()
        }
        planet_colors = {
            'Mercury': 'grey', 'Venus': 'goldenrod', 'Earth': 'green', 'Mars': 'red', 'Jupiter': 'orange',
            'Saturn': 'gold', 'Uranus': 'cyan', 'Neptune': 'blue', 'Pluto': 'darkgrey'
        }

        # Plot planets
        for name, planet in planets.items():
            if name == 'Earth':
                sun.compute(obs)
                ecl = ephem.Ecliptic(sun)
                theta = (ecl.lon + np.pi) % (2 * np.pi)
                r = 1  # Earth at 1 AU
            else:
                planet.compute(obs)
                theta = planet.hlon
                r = planet.sun_distance  # Linear scaling
            ax3.scatter(theta, r, color=planet_colors[name], s=50, label=name, zorder=2)

        # Asteroid belt as a single dashed line
        theta_values = np.linspace(0, 2 * np.pi, 500)
        ax3.plot(theta_values, [2.7] * 500, '--', color='purple', alpha=0.5, label='Asteroid Belt')

        # Voyager positions (realistic distances)
        ax3.scatter(np.radians(120), 125, color='black', s=20, label='Voyager 1', zorder=4)
        ax3.scatter(np.radians(240), 105, color='black', s=20, label='Voyager 2', marker='x', zorder=4)

        # Customize solar system plot
        ax3.set_ylim(0, 130)  # Extend to include Voyagers
        ax3.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.65, 1.0))
        ax3.set_title("Solar System Overview", fontsize=14, weight='bold', pad=15)
        ax3.set_yticklabels([])

        # Finalize layout and send plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        await ctx.send(file=discord.File(buf, 'astro_plot.png'))
        buf.close()
        plt.close(fig)

        logging.info(f"User {ctx.author} requested astronomy info for {location}")

    except (AttributeError, ValueError) as e:
        await ctx.send(f"Error retrieving astronomy information: {e}")

astro.help = """
**$astro [location] [time]**
Provides sunrise, sunset, moon phase, twilight info, and solar system overview.
**Arguments:**
- `location`: ICAO code or lat/lon pair (e.g., 'kmge' or '34.0522/-118.2437').
- `time` (optional): Time in 'HH:MM' format (e.g., '14:30'); defaults to current local time.
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

#############################
#############################
#############################
#MEDICINE STUFF
#############################
#############################
#############################
#
#############################
#############################
#############################
#BIBLE STUFF
#############################
#############################

import networkx as nx
import csv

# Define OT and NT books
OT_BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
            "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
            "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
            "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
            "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"]
NT_BOOKS = ["Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians",
            "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
            "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter",
            "1 John", "2 John", "3 John", "Jude", "Revelation"]

# Book abbreviations
BOOK_ABBREVIATIONS = {
    "Genesis": "Gen", "Exodus": "Exo", "Leviticus": "Lev", "Numbers": "Num", "Deuteronomy": "Deu",
    "Matthew": "Mat", "Mark": "Mar", "Luke": "Luk", "John": "Joh", "Acts": "Act"  # Add more as needed
}

# Load cross-references
def load_connections(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [(row[0], row[1]) for row in reader]

# Abbreviate verse references
def abbreviate_verse(verse):
    book, rest = verse.split(' ', 1)
    abbr = BOOK_ABBREVIATIONS.get(book, book)
    return f"{abbr} {rest}"

@bot.command()
async def bibleconnection(ctx, ot_book: str = None, nt_book: str = None):
    """Generate an image of connections between an OT book and an NT book."""
    # Check if both arguments are provided
    if ot_book is None or nt_book is None:
        await ctx.send("Please provide both an Old Testament book and a New Testament book. Usage: $bibleconnection <OT_book> <NT_book>")
        return

    # Validate book names
    if ot_book not in OT_BOOKS or nt_book not in NT_BOOKS:
        await ctx.send("Invalid book names. Use one OT and one NT book (e.g., Genesis Matthew).")
        return

    # Load and filter connections
    connections = load_connections('cross_references.csv')
    filtered_connections = [
        (source, target) for source, target in connections
        if (source.startswith(ot_book) and target.startswith(nt_book)) or
           (source.startswith(nt_book) and target.startswith(ot_book))
    ]

    if not filtered_connections:
        await ctx.send(f"No connections found between {ot_book} and {nt_book}.")
        return

    # Create directed graph
    G = nx.DiGraph()
    for source, target in filtered_connections:
        G.add_edge(source, target)

    # Assign bipartite attribute
    for node in G.nodes:
        G.nodes[node]['bipartite'] = 0 if node.startswith(ot_book) else 1

    # Set edge colors
    for u, v in G.edges:
        G[u][v]['color'] = 'red' if G.nodes[u]['bipartite'] == 0 else 'green'

    # Position nodes
    pos = nx.bipartite_layout(G, [node for node in G.nodes if G.nodes[node]['bipartite'] == 0])

    # Draw graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels={node: abbreviate_verse(node) for node in G.nodes},
            node_color=[('blue' if G.nodes[node]['bipartite'] == 0 else 'red') for node in G.nodes],
            edge_color=[G[u][v]['color'] for u, v in G.edges],
            font_size=8, node_size=500)

    # Save to buffer and send
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    await ctx.send(file=discord.File(buf, 'connections.png'))

# Initialize Realtime object
realtime_obj = tropycal_realtime.Realtime()

if __name__ == "__main__":
    bot.run(token=config.DISCORD_TOKEN)
