import re
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import io
import discord
from discord.ext import commands
from datetime import datetime, timezone
from astral import LocationInfo
from astral.sun import sun
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # Add this line
from matplotlib.patheffects import withStroke
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent if you need to read message content

# Create the bot object
bot = commands.Bot(command_prefix='$', intents=intents)

# Get city coordinates and associated airport codes
cities = {
    "Biloxi": (30.4103, -88.9245, "KBIX"),
    "Gulfport": (30.4073, -89.0701, "KGPT"),
    "Bay St. Louis": (30.3677, -89.4545, "KHSA"),
    "Mobile": (30.6267, -88.0681, "KBFM"),
    "Jackson": (32.3112, -90.0755, "KJAN"),
    "Baton Rouge": (30.5326, -91.1496, "KBTR"),
    "New Orleans": (29.9934, -90.2580, "KMSY")
}

def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)

        # Check if any METAR data was found at all
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        # Extract raw METAR observation
        raw_metar = json_data[0]['rawOb']

        # Exclude remarks by splitting at 'RMK'
        main_body = raw_metar.split('RMK')[0]

        # Extract temperature and dew point (in the form xx/yy or Mxx/Myy)
        temp_dew_pattern = re.search(r'\b(M?\d{2})/(M?\d{2})\b', main_body)
        if temp_dew_pattern:
            temp_str, dew_str = temp_dew_pattern.groups()
            temp_c = int(temp_str.replace('M', '-'))
            dew_point_c = int(dew_str.replace('M', '-'))
            temp_f = round((temp_c * 9 / 5) + 32)  # Convert to Fahrenheit
            dew_point_f = round((dew_point_c * 9 / 5) + 32)  # Convert to Fahrenheit
        else:
            temp_f = None
            dew_point_f = None

        # Extract wind direction and speed (in the form dddff or dddffGgg)
        wind_pattern = re.search(r'(\d{3})(\d{2})(G\d{2})?KT', main_body)
        if wind_pattern:
            wind_direction = int(wind_pattern.group(1))
            wind_speed = int(wind_pattern.group(2))
        else:
            wind_direction = None
            wind_speed = None

        # Return raw_metar along with other values
        return raw_metar, temp_f, dew_point_f, wind_direction, wind_speed

    except requests.exceptions.RequestException as e:
        # Handle network errors during fetching
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        # Handle potential parsing errors
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

def extract_cloud_info(metar):
    cloud_levels = {
        "low": [],
        "mid": [],
        "high": [],
        "vertical_visibility": None
    }

    # Exclude remarks by splitting at 'RMK'
    main_body = metar.split('RMK')[0]

    # Regular expression for matching cloud cover and vertical visibility
    cloud_pattern = re.compile(r'(FEW|SCT|BKN|OVC)(\d{3})|VV(\d{3})')

    # Search for cloud layers or vertical visibility in the main body
    cloud_matches = re.findall(cloud_pattern, main_body)

    for match in cloud_matches:
        if match[0]:  # Cloud layer (FEW, SCT, BKN, OVC)
            cover = match[0]
            altitude_hundreds = int(match[1])  # in hundreds of feet
            altitude_ft = altitude_hundreds * 100  # convert to feet

            # Categorize cloud levels based on altitude
            if altitude_ft <= 6500:
                cloud_levels["low"].append((cover, altitude_ft))
            elif 6500 < altitude_ft <= 20000:
                cloud_levels["mid"].append((cover, altitude_ft))
            else:
                cloud_levels["high"].append((cover, altitude_ft))

        if match[2]:  # Vertical visibility (VVxxx)
            vv_hundreds = int(match[2])
            cloud_levels["vertical_visibility"] = vv_hundreds * 100  # convert to feet

    return cloud_levels

def extract_weather_phenomena(metar):
    # Exclude remarks by splitting at 'RMK'
    main_body = metar.split('RMK')[0]

    # Regular expression to find weather phenomena in METAR
    weather_pattern = re.compile(r'(-|\+|VC)?(TS|SH|FZ|DR|BL|MI|BC|PR)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP)?(BR|FG|FU|VA|DU|SA|HZ|PY)?(PO|SQ|FC|SS|DS)?')

    # Split the main body of the METAR string into parts
    parts = main_body.split()
    weather_conditions = []

    for part in parts:
        match = re.match(weather_pattern, part)
        if match and any(match.groups()):
            weather_conditions.append(part)

    return weather_conditions

def map_metar_weather_to_condition(weather_codes, cloud_info, is_day):
    # Initialize the condition
    condition = None

    # Map weather codes to conditions
    if any('TS' in code for code in weather_codes):
        condition = 'Mostly Cloudy & Storming'
    elif any('RA' in code or 'SHRA' in code for code in weather_codes):
        condition = 'Mostly Cloudy & Rain' if cloud_info else 'Rain'
    elif any('SN' in code for code in weather_codes):
        condition = 'Snowing'
    elif any('FZRA' in code or 'PL' in code for code in weather_codes):
        condition = 'Wintry Mix'
    elif any('FG' in code or 'BR' in code for code in weather_codes):
        condition = 'Fog or Mist'
    elif cloud_info:
        # Determine cloudiness based on cloud cover
        covers = [layer[0] for layer in cloud_info]
        if any(cover in ['BKN', 'OVC'] for cover in covers):
            condition = 'Cloudy'
        elif any(cover in ['FEW', 'SCT'] for cover in covers):
            condition = 'Partially Cloudy (Sun)' if is_day else 'Partially Cloudy (Moon)'
        else:
            condition = 'Sunny' if is_day else 'Moonlight'
    else:
        condition = 'Sunny' if is_day else 'Moonlight'

    return condition

def is_daytime(lat, lon):
    city = LocationInfo(latitude=lat, longitude=lon)
    s = sun(city.observer, date=datetime.now(timezone.utc))
    now = datetime.now(timezone.utc)
    return s['sunrise'] <= now <= s['sunset']

weather_icons = {
    'Sunny': 'icons/sunny.png',
    'Moonlight': 'icons/moon.png',
    'Partially Cloudy (Sun)': 'icons/partly_cloudy.png',
    'Partially Cloudy (Moon)': 'icons/partly_moon.png',
    'Cloudy': 'icons/cloudy.png',
    'Mostly Cloudy & Rain': 'icons/partly_rain.png',
    'Mostly Cloudy & Storming': 'icons/storm.png',
    'Fog or Mist': 'icons/fog.png',
    'Snowing': 'icons/snow.png',
    'Wintry Mix': 'icons/wintry_mix.png',
    'Rain': 'icons/rain.png',
}

@bot.command(name='wrd_wx')
async def wrd_wx(ctx):
    async with ctx.typing():
        temperatures = {}
        dew_points = {}
        wind_directions = {}
        wind_speeds = {}
        weather_conditions = {}

        for city, coords in cities.items():
            lat, lon, icao = coords
            try:
                raw_metar, temp_f, dew_point_f, wind_direction, wind_speed = get_metar(icao)
                # Extract cloud info
                cloud_info = extract_cloud_info(raw_metar)
                # Extract weather phenomena
                weather_codes = extract_weather_phenomena(raw_metar)
                # Determine if it's day or night
                is_day = is_daytime(lat, lon)
                # Map METAR weather codes to condition
                cloud_layers = cloud_info['low'] + cloud_info['mid'] + cloud_info['high']
                condition = map_metar_weather_to_condition(weather_codes, cloud_layers, is_day)
                # Store data
                temperatures[city] = temp_f
                dew_points[city] = dew_point_f
                wind_directions[city] = wind_direction
                wind_speeds[city] = wind_speed
                weather_conditions[city] = condition
            except Exception as e:
                print(f"Error retrieving data for {city}: {e}")
                temperatures[city] = None
                dew_points[city] = None
                wind_directions[city] = None
                wind_speeds[city] = None
                weather_conditions[city] = None

        # Debug: Print retrieved data to ensure correctness
        print("Retrieved temperatures:", temperatures)
        print("Retrieved dew points:", dew_points)
        print("Retrieved wind directions and speeds:", wind_directions, wind_speeds)
        print("Retrieved weather conditions:", weather_conditions)

        # Create the map
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor('lightsteelblue')
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([-91.65, -87.57, 29.87, 32.81], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, zorder=0, facecolor='green', edgecolor='green')
        ax.add_feature(cfeature.COASTLINE, zorder=1, edgecolor='black')  # Keep coastline black for contrast
        ax.add_feature(cfeature.STATES, linestyle=":", zorder=1, linewidth=3)

        # Add interstates, US highways, and state highways
        roads_shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='roads')
        roads_reader = shpreader.Reader(roads_shpfilename)
        for record in roads_reader.records():
            if record.attributes['type'] in ['Major Highway', 'Secondary Highway']:
                ax.add_geometries(record.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='red' if record.attributes['type'] == 'Major Highway' else 'blue', linestyle='-', zorder=3, alpha=0.5)

        # Define colormap and normalization for temperature
        cmap = cm.get_cmap('jet')
        boundaries = np.linspace(-30, 120, 16)
        norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, extend='both')

        # Create a ScalarMappable and initialize it with the colormap and normalization
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Can pass temperatures.values() if needed

        # Plot city data
        for city, temp in temperatures.items():
            lat, lon, _ = cities[city]
            if temp is not None:
                # Plot scatter point for temperature using colormap
                color = cmap(norm(temp))
                ax.scatter(lon, lat, color=color, s=200, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree(), zorder=10)

            '''# Plot temperature in red
            ax.text(
                lon,
                lat + 0.2,  # Increase the offset for better readability
                f"{temp:.0f}°F",
                color='red',
                transform=ccrs.PlateCarree(),
                fontsize=14,  # Increased font size
                fontweight='bold',  # Bold font
                ha="center",
                va="bottom",
                zorder=12
            )

            # Plot dew point in green
            if dew_point is not None:
                ax.text(
                    lon,
                    lat - 0.2,  # Increase the offset for better readability
                    f"{dew_point:.0f}°F",
                    color='green',
                    transform=ccrs.PlateCarree(),
                    fontsize=14,  # Increased font size
                    ha="center",
                    va="top",
                    zorder=12
                )'''

            # Plot temperature with a background box for better readability if available
            temp = temperatures.get(city)
            if temp is not None:
                ax.text(
                    lon,
                    lat + 0.1,
                    f"{temp:.0f}°F",
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),  # Background box
                    transform=ccrs.PlateCarree(),
                    fontsize=14,
                    fontweight='bold',
                    ha="center",
                    va="bottom",
                    zorder=12
                )

            # Plot dew point in green if available
            dew_point = dew_points.get(city)
            if dew_point is not None:
                ax.text(
                    lon,
                    lat - 0.1,  # Increase the offset for better readability
                    f"{dew_point:.0f}°F",
                    color='green',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),  # Background box for better readability
                    transform=ccrs.PlateCarree(),
                    fontsize=14,  # Increased font size
                    ha="center",
                    va="top",
                    zorder=12
                )

            '''# outline muh numbers
            ax.text(
                lon,
                lat + 0.2,
                f"{temp:.0f}°F",
                color='red',
                fontsize=14,
                fontweight='bold',
                path_effects=[withStroke(linewidth=3, foreground='white')],  # White outline
                transform=ccrs.PlateCarree(),
                ha="center",
                va="bottom",
                zorder=12
            )'''

            # Plot wind barbs if wind data is available
            wind_direction = wind_directions.get(city)
            wind_speed = wind_speeds.get(city)
            if wind_direction is not None and wind_speed is not None:
                ax.barbs(
                    lon,
                    lat,
                    np.sin(np.radians(wind_direction)) * wind_speed, np.cos(np.radians(wind_direction)) * wind_speed,  # Convert speed and direction into components
                    color='purple',
                    transform=ccrs.PlateCarree(),
                    zorder=11
                )

            # Plot weather icon
            condition = weather_conditions.get(city)
            icon_path = weather_icons.get(condition)
            if icon_path and os.path.exists(icon_path):
                # Load the image
                image = plt.imread(icon_path)

                # Create an OffsetImage with increased size
                im = OffsetImage(image, zoom=0.3)  # Adjusted zoom for 2x larger size

                # Create an AnnotationBbox to place the icon to the right of the scatter plot
                ab = AnnotationBbox(
                    im,
                    (lon, lat),  # Position of the scatter plot point
                    xybox=(10, 0),  # Offset to move the icon to the right
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False,
                    box_alignment=(0, 0.25),  # Align to the center vertically
                    zorder=20
                )
                ax.add_artist(ab)
            else:
                print(f"Icon not found for condition '{condition}' in city '{city}'")

        # Re-add city names on the map
        for city, coords in cities.items():
            lat, lon, _ = coords
            ax.text(
                lon,
                lat,
                city,
                transform=ccrs.PlateCarree(),
                fontsize=12,
                ha="right",
                va="top",
                color='black',
                zorder=15,
            )

        # Add colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°F)', fontsize=12)

        # Add title
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M %p %Z')
        plt.title(f"Keesler AFB - Temperatures and Dew Points ({current_time})", fontsize=16, fontweight='bold')

        # Add legend for temperature, dew point, and icons
        # Create custom legend handles
        temp_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Temperature (°F)')
        dew_legend = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Dew Point (°F)')

        # Create patches for different weather conditions using the weather icons
        # sunny_patch = mpatches.Patch(color='yellow', label='Sunny')  # Add other icons similarly
        # cloudy_patch = mpatches.Patch(color='gray', label='Cloudy')
        # rain_patch = mpatches.Patch(color='blue', label='Rain')

        ax.legend(handles=[temp_legend, dew_legend], loc='upper right', fontsize=14)

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Send the image to the channel
        await ctx.send(file=discord.File(buf, filename="georgia_temp.png"))

        # Close the plot
        plt.close()
