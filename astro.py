# astro.py
import discord
from discord.ext import commands
import ephem
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime, timedelta
import pytz
from timezonefinder import TimezoneFinder
import airportsdata
import os
import logging
import json
from PIL import Image  # For handling images (logos)
import re
from matplotlib.patches import Circle, Wedge  # For moon phase visualization
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)


# Intents are required for Discord.py 1.5+
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='$', intents=intents)

# Define a custom converter for time (must be at the top level for import)
class TimeConverter(commands.Converter):
    async def convert(self, ctx, argument):
        try:
            hour, minute = map(int, argument.split(':'))
            return datetime(year=2024, month=1, day=1, hour=hour, minute=minute).time()
        except ValueError:
            raise commands.BadArgument("Invalid time format. Please use 'HH:MM' (e.g., '14:30').")

def get_metar(icao, hoursback=0, format='json'):
    """Fetches METARs for the specified airport code using your logic."""
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url, timeout=10).content
        json_data = json.loads(src)

        # Check if any METAR data was found at all
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        # Extract raw METAR observations
        raw_metars = [entry['rawOb'] for entry in json_data if 'rawOb' in entry]
        if not raw_metars:
            raise ValueError(f"No raw METAR observations found for {icao}.")

        # Ensure we return a single string (most recent)
        latest_metar = max(raw_metars, key=lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else '000000Z')
        logging.info(f"Successfully fetched METAR for {icao}: {latest_metar}")
        return latest_metar  # Return string, not list

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching METAR data for {icao}: {e}")
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Error parsing METAR data for {icao}: {e}")
        return None

def parse_metar(metar_string):
    """Parse METAR string for weather data including temperature, dew point, visibility, and sky conditions."""
    if not metar_string:
        return "Weather data unavailable"

    # Extract temperature and dew point from the METAR string (e.g., 06/M02)
    temp_dew_match = re.search(r"(\d{2})/([M]?\d{2})", metar_string)
    temp_c, dew_c = None, None
    if temp_dew_match:
        temp_f = int(temp_dew_match.group(1))  # Temperature in Fahrenheit
        temp_c = (temp_f - 32) / 1.8  # Convert to Celsius
        dew_str = temp_dew_match.group(2)
        dew_sign = -1 if dew_str.startswith('M') else 1
        dew_f = dew_sign * int(dew_str.replace('M', ''))  # Dew point in Fahrenheit
        dew_c = (dew_f - 32) / 1.8  # Convert to Celsius

    # Visibility (in statute miles)
    visibility_match = re.search(r"(\d+)SM", metar_string)
    visibility = visibility_match.group(1) if visibility_match else "10"  # Default to 10 SM for clear skies if not specified

    # Sky conditions (e.g., BKN200 OVC230)
    sky_conditions = re.search(r"(FEW|SKC|SCT|BKN|OVC)\d{3}(?: (FEW|SKC|SCT|BKN|OVC)\d{3})*(?: (FEW|SKC|SCT|BKN|OVC)\d{3})?", metar_string)
    sky = sky_conditions.group(0) if sky_conditions else "Clear"

    # Construct weather string
    weather_str = "Weather: "
    #if temp_c is not None:
        #weather_str += f"Temp: {temp_c:.1f}°C, "
    #if dew_c is not None:
        #weather_str += f"Dew Point: {dew_c:.1f}°C, "
    weather_str += f"Visibility: {visibility} SM, Sky: {sky}"
    return weather_str.strip(", ")

def get_city_state(lat, lon, icao=None):
    """Get city and state for a location based on ICAO code or coordinates."""
    if icao:
        airports = airportsdata.load('icao')
        airport = airports.get(icao.upper())
        if airport:
            city = airport.get('city', 'Unknown')
            state = airport.get('state', 'Unknown')
            return f"{city}, {state}"
    # For lat/lon, use a simple reverse geocoding (fallback, can be enhanced with geopy)
    # This is a basic approximation; you might want to use geopy for more accuracy
    return "Unknown Location"

def draw_moon_phase(ax, phase_percentage):
    """Draw a simple lunar phase icon as a static inset in the plot."""
    from matplotlib.patches import Circle, Wedge  # Ensure Circle and Wedge are imported here
    ax_inset = ax.inset_axes([0.02, 0.02, 0.1, 0.1])  # Small inset in bottom-left
    circle = Circle((0.5, 0.5), 0.5, facecolor='gray', edgecolor='black')
    ax_inset.add_patch(circle)
    if phase_percentage < 50:  # Waning (left side illuminated)
        wedge = Wedge((0.5, 0.5), 0.5, 0, 180 * (phase_percentage / 50), facecolor='white')
    else:  # Waxing (right side illuminated)
        wedge = Wedge((0.5, 0.5), 0.5, 180 * ((phase_percentage - 50) / 50), 180, facecolor='white')
    ax_inset.add_patch(wedge)
    ax_inset.set_aspect('equal')
    ax_inset.axis('off')

# Define the astro command as a cog-like structure (since it's a standalone command, we'll make it a function)
async def astro_command(ctx, location: str = None, time: TimeConverter = None):
    """Provides sunrise, sunset, moon phase, twilight info, and solar system overview for a given location and time."""
    if not location:
        await ctx.send("Please provide an ICAO code or latitude/longitude pair (e.g., '$astro kmge' or '$astro 34.05/-118.25')")
        return

    try:
        icao = None
        if '/' in location:  # Check if input is a latitude/longitude pair
            try:
                lat, lon = map(float, location.split('/'))
                lat = round(lat, 4)
                lon = round(lon, 4)
            except ValueError:
                raise ValueError("Invalid latitude/longitude format. Please use 'lat/lon' (e.g., '34.0500/-118.2500').")
        else:
            # Get airport data using ICAO code
            icao = location.upper()
            airports = airportsdata.load('icao')
            airport = airports.get(icao)
            if not airport:
                raise ValueError("Airport not found.")

            # Extract latitude and longitude
            lat = round(airport['lat'], 4)
            lon = round(airport['lon'], 4)

        # Determine time zone (use lat, lon from airport data)
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lng=lon, lat=lat)

        if time:
            # Create a datetime object with the specified hour and minute in the local timezone
            local_time = datetime.now(pytz.timezone(timezone)).replace(hour=time.hour, minute=time.minute, second=0, microsecond=0)
        else:
            # Use the current local time
            local_time = datetime.now(pytz.timezone(timezone))

        # Convert the local time to UTC for PyEphem (9:10 PM EST Feb 23, 2025 = 02:10 UTC Feb 24, 2025)
        now = local_time.astimezone(pytz.utc)

        now_local = now.astimezone(pytz.timezone(timezone))  # Convert UTC time to local time for display

        # Define observer before setting date
        obs = ephem.Observer()

        # Assuming local_time is defined earlier, and we convert it to UTC for PyEphem
        now = local_time.astimezone(pytz.utc)

        # Create a single figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))  # Create one combined figure
        fig.patch.set_facecolor('lightsteelblue')

        # Create grid layout for subplots
        gs = GridSpec(2, 1, height_ratios=[2, 1])  # Top plot takes more height than the bottom plot

        # --- Sun and Moon Plot (Top Plot) ---
        ax1 = fig.add_subplot(gs[0])

        # Calculate sunrise, sunset, twilight, etc.
        obs.lat = str(lat)  # Use latitude
        obs.lon = str(lon)  # Use longitude
        obs.date = now

        sun = ephem.Sun()
        moon = ephem.Moon()

        # Sunrise and Sunset Calculations
        sunrise = ephem.localtime(obs.next_rising(sun))
        sunset = ephem.localtime(obs.next_setting(sun))

        # Adjust sunrise and sunset for the Southern Hemisphere (keeping as is, per your request)
        is_southern_hemisphere = float(lat) < 0

        if is_southern_hemisphere:
            # Swap sunrise and sunset if in the Southern Hemisphere
            sunrise, sunset = sunset, sunrise

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

        # Generate Sun Path Plot
        ax1.patch.set_facecolor('white')
        times = [now + timedelta(minutes=15 * i) for i in range(96)]

        # Define observer and sun/moon
        obs.lat = str(lat)
        obs.lon = str(lon)
        sun = ephem.Sun()
        moon = ephem.Moon()

        # --- Sun Path (Modified to match your existing logic) ---
        # List to store Sun path segments
        sun_segments = []
        current_sun_segment = []
        current_sun_linestyle = None

        # Iterate through each time interval to compute sun position
        for i, time in enumerate(times):
            obs.date = time
            sun.compute(obs)
            azimuth = np.degrees(sun.az)
            altitude = np.degrees(sun.alt)

            # Handle azimuth wrapping and line style based on altitude
            if current_sun_segment:
                prev_azimuth, prev_altitude = current_sun_segment[-1]

                # Check for large azimuth or altitude jump to prevent wraparound lines
                if abs(azimuth - prev_azimuth) > 180 or abs(altitude - prev_altitude) > 45:
                    sun_segments.append((current_sun_segment, current_sun_linestyle))
                    current_sun_segment = []

            # Determine linestyle based on altitude (dashed below horizon, solid above)
            sun_linestyle = '--' if altitude <= 0 else '-'

            # If linestyle changes, save the current segment and start a new one
            if current_sun_linestyle is not None and sun_linestyle != current_sun_linestyle:
                sun_segments.append((current_sun_segment, current_sun_linestyle))
                current_sun_segment = []

            current_sun_linestyle = sun_linestyle
            current_sun_segment.append((azimuth, altitude))

        # Append the last Sun segment if it's not empty
        if current_sun_segment:
            sun_segments.append((current_sun_segment, current_sun_linestyle))

        # Plot Sun path segments
        for segment, linestyle in sun_segments:
            if len(segment) < 2:
                continue  # Skip segments that are too small

            azimuths, altitudes = zip(*segment)
            azimuths = [az % 360 for az in azimuths]  # Adjust azimuths back to the 0-360 range

            # Plot only if azimuths are continuous
            if max(azimuths) - min(azimuths) < 180:
                ax1.plot(azimuths, altitudes, color='orange', lw=3, linestyle=linestyle, label='Sun Path' if linestyle == '-' else "")

        # --- Moon Path (New, similar to Sun path) ---
        # List to store Moon path segments
        moon_segments = []
        current_moon_segment = []
        current_moon_linestyle = None

        # Iterate through each time interval to compute moon position
        for i, time in enumerate(times):
            obs.date = time
            moon.compute(obs)
            azimuth = np.degrees(moon.az)
            altitude = np.degrees(moon.alt)

            # Handle azimuth wrapping and line style based on altitude
            if current_moon_segment:
                prev_azimuth, prev_altitude = current_moon_segment[-1]

                # Check for large azimuth or altitude jump to prevent wraparound lines
                if abs(azimuth - prev_azimuth) > 180 or abs(altitude - prev_altitude) > 45:
                    moon_segments.append((current_moon_segment, current_moon_linestyle))
                    current_moon_segment = []

            # Determine linestyle based on altitude (dashed below horizon, solid above)
            moon_linestyle = '--' if altitude <= 0 else '-'

            # If linestyle changes, save the current segment and start a new one
            if current_moon_linestyle is not None and moon_linestyle != current_moon_linestyle:
                moon_segments.append((current_moon_segment, current_moon_linestyle))
                current_moon_segment = []

            current_moon_linestyle = moon_linestyle
            current_moon_segment.append((azimuth, altitude))

        # Append the last Moon segment if it's not empty
        if current_moon_segment:
            moon_segments.append((current_moon_segment, current_moon_linestyle))

        # Plot Moon path segments
        for segment, linestyle in moon_segments:
            if len(segment) < 2:
                continue  # Skip segments that are too small

            azimuths, altitudes = zip(*segment)
            azimuths = [az % 360 for az in azimuths]  # Adjust azimuths back to the 0-360 range

            # Plot only if azimuths are continuous
            if max(azimuths) - min(azimuths) < 180:
                ax1.plot(azimuths, altitudes, color='gray', lw=3, linestyle=linestyle, label='Moon Path' if linestyle == '-' else "")

        # Plot current sun and moon positions (keeping gray circle for Moon)
        obs.date = now
        sun.compute(obs)
        moon.compute(obs)

        current_sun_az = np.degrees(sun.az)
        current_sun_alt = np.degrees(sun.alt)

        current_moon_az = np.degrees(moon.az)
        current_moon_alt = np.degrees(moon.alt)

        # Plot twilight periods along y-axis
        ax1.axhspan(-90, -12, color='midnightblue', alpha=0.3, label='Astronomical Twilight')
        ax1.axhspan(-12, -6, color='deepskyblue', alpha=0.3, label='Nautical Twilight')
        ax1.axhspan(-6, -0.34, color='lightskyblue', alpha=0.3, label='Civil Twilight')
        # Shade nighttime
        ax1.axhspan(-90, -18, color='black', alpha=0.5, label='Night')

        # Mark the current position of the sun
        ax1.scatter(current_sun_az, current_sun_alt, color='yellow', edgecolors='black', s=150, label='Current Sun Position', zorder=3)

        # Mark the current position of the moon (keeping gray circle)
        ax1.scatter(current_moon_az, current_moon_alt, color='gray', edgecolors='black', s=150, label='Current Moon Position', zorder=3)

        # Horizon Line
        ax1.axhline(0, color='blue', linestyle='--', lw=1.5, label='Horizon')

        # Add text with ALL astronomy information to the plot, including weather and city/state
        moon_phase_text = f"Moon Phase: {moon_phase:.1f}% illuminated"
        if moon.phase < 50:
            moon_phase_text += " (Waning)"
        else:
            moon_phase_text += " (Waxing)"

        # Calculate solar noon
        now = datetime.utcnow()  # Get current UTC time (9:10 PM EST Feb 23, 2025 = 02:10 UTC Feb 24, 2025)
        obs.date = now.strftime('%Y/%m/%d %H:%M:%S')  # Set observer date to current time or user-specified time
        solar_noon = ephem.localtime(obs.next_transit(sun))  # Calculate solar noon time

        # Calculate solar altitude at solar noon
        obs.date = solar_noon
        sun.compute(obs)
        solar_noon_altitude = np.degrees(sun.alt)

        # Determine optimal solar panel angle (basic example)
        lat_deg = np.degrees(float(obs.lat))  # Get observer's latitude in degrees
        optimal_angle = lat_deg  # Use latitude as a basic approximation for optimal solar panel angle

        # Get weather data (fetch real-time METAR for ICAO or fallback for lat/lon)
        if icao:
            metar_data = get_metar(icao, hoursback=1)  # Use your get_metar with 1 hour back for recent data
            logging.info(f"metar_data for {icao}: {metar_data}")
        else:
            # For lat/lon, approximate with nearest airport (simplified; could use geopy for better accuracy)
            metar_data = "METAR data unavailable for coordinates"  # Placeholder for lat/lon

        weather_info = parse_metar(metar_data) if isinstance(metar_data, str) and metar_data != "METAR data unavailable" else "Weather data unavailable"
        logging.info(f"Input to parse_metar: {metar_data}")

        # Get city and state
        location_str = location.upper() if '/' not in location else f"Lat: {lat}, Lon: {lon}"
        city_state = get_city_state(lat, lon, icao if icao else None)

        # Ensure textstr fits in the tan box with smaller font if needed
        textstr = f"Location: {location_str} - {city_state}\n" \
                  f"{weather_info}\n" \
                  f"Sun Azimuth: {current_sun_az:.1f}°\n" \
                  f"Moon Azimuth: {current_moon_az:.1f}°\n" \
                  f"Moon Elevation: {current_moon_alt:.1f}°\n" \
                  f"{moon_phase_text}\n\n" \
                  f"Sunrise: {sunrise.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Sunset: {sunset.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Civil Twilight Begin: {civil_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Civil Twilight End: {civil_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Nautical Twilight Begin: {nautical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Nautical Twilight End: {nautical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Astronomical Twilight Begin: {astronomical_twilight_begin.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Astronomical Twilight End: {astronomical_twilight_end.strftime('%Y-%m-%d %I:%M %p %Z')}\n" \
                  f"Optimal Solar Panel Angle: ({optimal_angle:.1f}°)"

        # These are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

        # Use plt.figtext to place the text, ensuring it’s large enough to show all lines
        plt.figtext(0.05, 0.05, textstr, fontsize=9, weight='bold', va="bottom", bbox=props)  # Moved lower and kept font size

        # Add moon phase visualization
        draw_moon_phase(ax1, moon_phase)

        # Labels and title
        ax1.set_title(f"Sun/Moon Position: {location.upper() if '/' not in location else f'Lat: {lat}, Lon: {lon}'} - {lat:.4f}, {lon:.4f}\n{now_local.strftime('%b %d %Y %H:%M')} Local Time", fontsize=14, weight='bold')
        ax1.set_xlabel("Azimuth (degrees)", weight='bold')
        ax1.set_ylabel("Altitude (degrees)", weight='bold')
        ax1.set_xlim(0, 360)
        ax1.set_ylim(-90, 90)

        # Customize x-axis to show BOTH cardinal directions and degrees
        ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax1.set_xticklabels(['N\n0°', 'NE\n45°', 'E\n90°', 'SE\n135°', 'S\n180°', 'SW\n225°', 'W\n270°', 'NW\n315°', 'N\n360°'])

        # Annotate the current sun position with specific cardinal direction and azimuth angle
        if 348.75 <= current_sun_az <= 360 or 0 <= current_sun_az < 11.25:
            sun_direction = "N"
        elif 11.25 <= current_sun_az < 33.75:
            sun_direction = "NNE"
        elif 33.75 <= current_sun_az < 56.25:
            sun_direction = "NE"
        elif 56.25 <= current_sun_az < 78.75:
            sun_direction = "ENE"
        elif 78.75 <= current_sun_az < 101.25:
            sun_direction = "E"
        elif 101.25 <= current_sun_az < 123.75:
            sun_direction = "ESE"
        elif 123.75 <= current_sun_az < 146.25:
            sun_direction = "SE"
        elif 146.25 <= current_sun_az < 168.75:
            sun_direction = "SSE"
        elif 168.75 <= current_sun_az < 191.25:
            sun_direction = "S"
        elif 191.25 <= current_sun_az < 213.75:
            sun_direction = "SSW"
        elif 213.75 <= current_sun_az < 236.25:
            sun_direction = "SW"
        elif 236.25 <= current_sun_az < 258.75:
            sun_direction = "WSW"
        elif 258.75 <= current_sun_az < 281.25:
            sun_direction = "W"
        elif 281.25 <= current_sun_az < 303.75:
            sun_direction = "WNW"
        elif 303.75 <= current_sun_az < 326.25:
            sun_direction = "NW"
        else:  # 326.25 <= current_sun_az < 348.75
            sun_direction = "NNW"

        # Annotate the current sun position
        ax1.annotate(f"{sun_direction} (Cardinal Direction)\nAzimuth: ({current_sun_az:.1f}°)\nElevation: ({current_sun_alt:.1f}°)",
                    (current_sun_az, current_sun_alt), textcoords="offset points",
                    xytext=(10, 10), ha='center', fontsize=10, color='darkred', fontweight='bold')

        # Annotate the current moon position with cardinal direction, azimuth, and altitude
        if 0 <= current_moon_az < 45 or 315 <= current_moon_az <= 360:
            moon_direction = "N"
        elif 45 <= current_moon_az < 135:
            moon_direction = "E"
        elif 135 <= current_moon_az < 225:
            moon_direction = "S"
        else:
            moon_direction = "W"

        ax1.annotate(f"{moon_direction} (Cardinal Direction)\nAzimuth: ({current_moon_az:.1f})\nElevation: ({current_moon_alt:.1f}°)",
                    (current_moon_az, current_moon_alt), textcoords="offset points",
                    xytext=(10, 10), ha='center', fontsize=10, color='black', fontweight='bold')

        # Load and resize the icons, multiplying the height scaling by 3 for a larger size
        dpi = fig.dpi
        quarter_inch_in_pixels = 0.25 * dpi * 4  # 3x larger than original quarter inch height

        try:
            uga_logo = Image.open('/home/evanl/Documents/bot/Georgia_Bulldogs_logo.png')
            photo = Image.open('/home/evanl/Documents/bot/boxlogo2.png')

            uga_logo_resized = uga_logo.resize((int(quarter_inch_in_pixels * uga_logo.width / uga_logo.height), int(quarter_inch_in_pixels)))
            photo_resized = photo.resize((int(quarter_inch_in_pixels * photo.width / photo.height), int(quarter_inch_in_pixels)))

            # Add the resized icons to the plot
            ax1.figure.figimage(uga_logo_resized, 10, fig.bbox.ymax - uga_logo_resized.height - 10, zorder=1)
            ax1.figure.figimage(photo_resized, fig.bbox.xmax - photo_resized.width - 10, fig.bbox.ymax - photo_resized.height - 10, zorder=1)
        except FileNotFoundError as e:
            logging.error(f"Could not load image files: {e}")
            # Optionally, skip adding logos if files aren't found, or prompt user in Discord
            await ctx.send("Warning: Could not load UGA or box logos due to missing files.")

        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- Solar System Overview (Bottom Plot) ---
        ax3 = fig.add_subplot(gs[1], projection='polar')

        # Plot the Sun at the center
        ax3.scatter(0, 0, color='yellow', s=100, edgecolor='black', label='Sun', zorder=3)

        # Define planetary objects (including Earth)
        planets = {
            'Mercury': ephem.Mercury(),
            'Venus': ephem.Venus(),
            'Earth': 'Earth',  # Placeholder for Earth
            'Mars': ephem.Mars(),
            'Jupiter': ephem.Jupiter(),
            'Saturn': ephem.Saturn(),
            'Uranus': ephem.Uranus(),
            'Neptune': ephem.Neptune(),
            'Pluto': ephem.Pluto(),
        }

        planet_colors = {
            'Mercury': 'grey',
            'Venus': 'goldenrod',
            'Earth': 'green',  # Color for Earth
            'Mars': 'red',
            'Jupiter': 'orange',
            'Saturn': 'gold',
            'Uranus': 'cyan',
            'Neptune': 'blue',
            'Pluto': 'darkgrey',
        }

        # Plot each planet
        for name, planet in planets.items():
            if name == 'Earth':
                # Compute Earth's heliocentric longitude
                sun.compute(obs)
                earth_hlon = (sun.hlon + np.pi) % (2 * np.pi)  # Earth's heliocentric longitude is opposite the Sun's
                r = 1  # Earth's average distance from the Sun is 1 AU
                theta = earth_hlon
            else:
                planet.compute(obs)
                r = 1 + np.log10(planet.sun_distance)  # Use log to keep distances reasonable
                theta = planet.hlon  # Longitude of planet in radians
            ax3.scatter(theta, r, color=planet_colors[name], s=50, label=name, zorder=2)

        # Plot asteroid belt as a ring
        asteroid_belt_radius = 1.7  # Adjusted radius to be between Mars and Jupiter
        ax3.plot(np.linspace(0, 2 * np.pi, 500), [asteroid_belt_radius] * 500, '--', color='purple', alpha=0.5, label='Asteroid Belt')

        # Plot Voyager 1 and Voyager 2 positions
        voyager_1_radius = 3.5
        voyager_2_radius = 3.3
        voyager_1_theta = np.radians(120)
        voyager_2_theta = np.radians(240)

        ax3.scatter(voyager_1_theta, voyager_1_radius, color='black', s=20, label='Voyager 1', zorder=4)
        ax3.scatter(voyager_2_theta, voyager_2_radius, color='black', s=20, label='Voyager 2', marker='x', zorder=4)

        # Customize the solar system plot
        ax3.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.65, 1.0))  # Adjusted legend position
        ax3.set_title("Solar System Overview", fontsize=14, weight='bold', pad=15)
        ax3.set_ylim(0, 4)
        ax3.set_yticklabels([])

        # Adjust the layout of the figure to accommodate both plots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)  # Adjust space between subplots for clarity

        # Save the combined plot to a file
        combined_plot_filename = "/home/evanl/Documents/combined_sun_moon_solar_system_plot.png"
        plt.savefig(combined_plot_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Send the combined plot via Discord
        await ctx.send(file=discord.File(combined_plot_filename))

        logging.info(f"User {ctx.author} requested astronomy information for {location}")

    except (AttributeError, ValueError, requests.RequestException) as e:
        await ctx.send(f"Error retrieving astronomy information: {e}")

# Help text for the astro command
astro_help = """
**$astro [location]**

Provides sunrise, sunset, moon phase, and twilight information for a given location.

**Arguments:**

*   `location` (optional): The location for which you want to retrieve astronomy information. You can provide an ICAO airport code, or a latitude/longitude pair (e.g., '34.0522/-118.2437' for four decimal places, which represent the ten-thousandth digit).
"""
