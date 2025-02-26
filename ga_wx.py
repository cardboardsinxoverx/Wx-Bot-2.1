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
from datetime import datetime, timezone
from astral import LocationInfo
from astral.sun import sun
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import discord  # Needed for discord.File
import asyncio

# Define station-to-city mapping
cities = {
    "Atlanta": (33.7490, -84.3880, "KATL"),
    "Savannah": (32.0809, -81.0912, "KSAV"),
    "Athens": (33.9519, -83.3576, "KAHN"),
    "Rome": (34.2570, -85.1647, "KRMG"),
    "Dalton": (34.7698, -84.9702, "KDNN"),
    "Gainesville": (34.2979, -83.8241, "KGVL"),
    "Clayton": (34.8781, -83.4011, "K1A5"),
    "Valdosta": (30.8327, -83.2785, "KVLD"),
    "Albany": (31.5785, -84.1557, "KABY"),
    "Waycross": (31.2136, -82.3549, "KAYS"),
    "Robins AFB": (32.6400, -83.5917, "KWRB"),
}

# Helper functions (unchanged from your original)
def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")
        raw_metar = json_data[0]['rawOb']
        main_body = raw_metar.split('RMK')[0]
        temp_dew_pattern = re.search(r'\b(M?\d{2})/(M?\d{2})\b', main_body)
        if temp_dew_pattern:
            temp_str, dew_str = temp_dew_pattern.groups()
            temp_c = int(temp_str.replace('M', '-'))
            dew_point_c = int(dew_str.replace('M', '-'))
            temp_f = round((temp_c * 9 / 5) + 32)
            dew_point_f = round((dew_point_c * 9 / 5) + 32)
        else:
            temp_f = None
            dew_point_f = None
        wind_pattern = re.search(r'(\d{3})(\d{2})(G\d{2})?KT', main_body)
        if wind_pattern:
            wind_direction = int(wind_pattern.group(1))
            wind_speed = int(wind_pattern.group(2))
        else:
            wind_direction = None
            wind_speed = None
        altimeter_inHg = None
        alt_pattern_us = re.search(r'\bA(\d{4})\b', main_body)
        if alt_pattern_us:
            alt_raw = alt_pattern_us.group(1)
            altimeter_inHg = float(alt_raw) / 100.0
        else:
            alt_pattern_q = re.search(r'\bQ(\d{4})\b', main_body)
            if alt_pattern_q:
                alt_raw = alt_pattern_q.group(1)
                alt_hpa = float(alt_raw)
                altimeter_inHg = round(alt_hpa * 0.02953, 2)
        return raw_metar, temp_f, dew_point_f, wind_direction, wind_speed, altimeter_inHg
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

def extract_cloud_info(metar):
    cloud_levels = {"low": [], "mid": [], "high": [], "vertical_visibility": None}
    main_body = metar.split('RMK')[0]
    cloud_pattern = re.compile(r'(FEW|SCT|BKN|OVC)(\d{3})|VV(\d{3})')
    cloud_matches = re.findall(cloud_pattern, main_body)
    for match in cloud_matches:
        if match[0]:
            cover = match[0]
            altitude_hundreds = int(match[1])
            altitude_ft = altitude_hundreds * 100
            if altitude_ft <= 6500:
                cloud_levels["low"].append((cover, altitude_ft))
            elif 6500 < altitude_ft <= 20000:
                cloud_levels["mid"].append((cover, altitude_ft))
            else:
                cloud_levels["high"].append((cover, altitude_ft))
        if match[2]:
            vv_hundreds = int(match[2])
            cloud_levels["vertical_visibility"] = vv_hundreds * 100
    return cloud_levels

def extract_weather_phenomena(metar):
    main_body = metar.split('RMK')[0]
    weather_pattern = re.compile(r'(-|\+|VC)?(TS|SH|FZ|DR|BL|MI|BC|PR)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP)?(BR|FG|FU|VA|DU|SA|HZ|PY)?(PO|SQ|FC|SS|DS)?')
    parts = main_body.split()
    weather_conditions = []
    for part in parts:
        match = re.match(weather_pattern, part)
        if match and any(match.groups()):
            weather_conditions.append(part)
    return weather_conditions

def map_metar_weather_to_condition(weather_codes, cloud_info, is_day):
    condition = None
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
    loc = LocationInfo(latitude=lat, longitude=lon)
    s = sun(loc.observer, date=datetime.now(timezone.utc))
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

def get_nws_alerts(state='GA'):
    url = f"https://api.weather.gov/alerts/active?area={state}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36")
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "features" not in data:
            return []
        alerts = []
        for feature in data["features"]:
            props = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            affected_areas = props.get("areaDesc", "Unknown Area")
            if geometry:
                if geometry.get("type") == "Polygon":
                    coords = geometry["coordinates"][0]
                elif geometry.get("type") == "MultiPolygon":
                    coords = geometry["coordinates"][0][0]
                else:
                    coords = []
            else:
                coords = []
            if coords:
                lons, lats = zip(*coords)
            else:
                lons, lats = ([], [])
            alerts.append({
                "headline": props.get("headline", "No headline"),
                "severity": props.get("severity", "Unknown"),
                "lons": np.array(lons),
                "lats": np.array(lats),
                "color": 'red' if props.get("severity") == "Extreme" else 'orange',
                "affected_areas": affected_areas
            })
        return alerts
    except requests.RequestException as e:
        print(f"Error fetching NWS alerts: {e}")
        return []

# Background task for weather map generation
async def generate_weather_map(ctx):
    """Generate and send the Georgia weather map in the background."""
    temperatures = {}
    dew_points = {}
    wind_directions = {}
    wind_speeds = {}
    altimeters = {}
    weather_conditions = {}

    # Fetch METAR data
    for city, coords in cities.items():
        lat, lon, icao = coords
        try:
            (raw_metar, temp_f, dew_point_f, wind_dir, wind_spd, alt_inHg) = get_metar(icao)
            cloud_info = extract_cloud_info(raw_metar)
            weather_codes = extract_weather_phenomena(raw_metar)
            day_flag = is_daytime(lat, lon)
            combined_cloud_layers = cloud_info['low'] + cloud_info['mid'] + cloud_info['high']
            condition = map_metar_weather_to_condition(weather_codes, combined_cloud_layers, day_flag)
            temperatures[city] = temp_f
            dew_points[city] = dew_point_f
            wind_directions[city] = wind_dir
            wind_speeds[city] = wind_spd
            altimeters[city] = alt_inHg
            weather_conditions[city] = condition
        except Exception as e:
            print(f"Error retrieving data for {city} ({icao}): {e}")
            temperatures[city] = None
            dew_points[city] = None
            wind_directions[city] = None
            wind_speeds[city] = None
            altimeters[city] = None
            weather_conditions[city] = None

    # Fetch NWS alerts
    alerts = get_nws_alerts('GA')

    # Plotting
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('lightsteelblue')
    title_ax = fig.add_axes([0, 0.92, 1, 0.08], facecolor='lightsteelblue')
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    title_ax.spines['top'].set_visible(False)
    title_ax.spines['right'].set_visible(False)
    title_ax.spines['bottom'].set_visible(False)
    title_ax.spines['left'].set_visible(False)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-85.7, -80.5, 30.2, 35.0], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='wheat', edgecolor='green')
    ax.add_feature(cfeature.COASTLINE, zorder=1, edgecolor='black')
    ax.add_feature(cfeature.STATES, linestyle=":", zorder=1, linewidth=3)

    # Optimize road plotting by pre-filtering records
    roads_shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='roads')
    roads_reader = shpreader.Reader(roads_shpfilename)
    relevant_roads = [
        record.geometry for record in roads_reader.records()
        if record.attributes['type'] in ['Major Highway', 'Secondary Highway']
        and record.geometry.bounds[0] <= -80.5  # East bound
        and record.geometry.bounds[2] >= -85.7  # West bound
        and record.geometry.bounds[1] <= 35.0   # North bound
        and record.geometry.bounds[3] >= 30.2   # South bound
    ]
    for geometry in relevant_roads:
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='none',
                          edgecolor='red' if 'Major Highway' in record.attributes['type'] else 'blue',
                          linestyle='-', zorder=3, alpha=0.5)

    cmap = cm.get_cmap('jet')
    boundaries = np.linspace(-30, 120, 16)
    norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, extend='both')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for city, temp in temperatures.items():
        lat, lon, _ = cities[city]
        if temp is not None:
            color = cmap(norm(temp))
            ax.scatter(lon, lat, color=color, s=150, edgecolor='black', linewidth=0.5,
                     transform=ccrs.PlateCarree(), zorder=10)
        ax.text(lon, lat + 0.2, city, transform=ccrs.PlateCarree(), fontsize=12,
                ha="center", va="bottom", color='black', zorder=15)
        if temp is not None:
            ax.text(lon, lat + 0.12, f"{temp:.0f}°F", color='red',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                    transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold',
                    ha="center", va="bottom", zorder=12)
        dew_point = dew_points.get(city)
        if dew_point is not None:
            ax.text(lon, lat - 0.06, f"DP: {dew_point:.0f}°F", color='green',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                    transform=ccrs.PlateCarree(), fontsize=12, ha="center", va="top", zorder=12)
        alt_inHg = altimeters.get(city)
        if alt_inHg is not None:
            ax.text(lon, lat - 0.14, f"Alt: {alt_inHg:.2f}\"Hg", color='blue',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                    transform=ccrs.PlateCarree(), fontsize=11, ha="center", va="top", zorder=12)
        wind_dir = wind_directions.get(city)
        wind_spd = wind_speeds.get(city)
        if wind_dir is not None and wind_spd is not None:
            rad_dir = np.radians(wind_dir)
            u = wind_spd * np.sin(np.pi/2 - rad_dir)
            v = wind_spd * np.cos(np.pi - rad_dir)
            ax.barbs(lon, lat, u, v, color='purple', edgecolor='black',
                    transform=ccrs.PlateCarree(), zorder=11)
            ax.text(lon + 0.3, lat, f"{wind_dir}°@{wind_spd}kt", color='purple',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                    transform=ccrs.PlateCarree(), fontsize=10, ha="left", va="center", zorder=12)
        condition = weather_conditions.get(city)
        icon_path = weather_icons.get(condition)
        if icon_path and os.path.exists(icon_path):
            img = plt.imread(icon_path)
            im = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(im, (lon, lat), xybox=(20, 0), xycoords='data',
                              boxcoords="offset points", frameon=False,
                              box_alignment=(0, 0.5), zorder=20)
            ax.add_artist(ab)

    for alert in alerts:
        if alert['lons'].size > 0 and alert['lats'].size > 0:
            ax.plot(alert['lons'], alert['lats'], color=alert['color'], linewidth=2,
                    transform=ccrs.PlateCarree(), zorder=21)
            center_lon = np.mean(alert['lons'])
            center_lat = np.mean(alert['lats'])
            ax.text(center_lon, center_lat, f"{alert['severity']}\n{alert['headline']}",
                    color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
                    transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', zorder=22)

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°F)', fontsize=12)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M %p %Z')
    title_ax.text(0.5, 0.5, f"Georgia - Detailed Weather Map ({current_time})",
                 fontsize=16, fontweight="bold", ha="center", va="center",
                 transform=title_ax.transAxes)

    temp_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                              markersize=10, label='Temperature (°F)')
    dew_legend = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                             markersize=10, label='Dew Point (°F)')
    alt_legend = mpatches.Patch(color='blue', label='Altimeter')
    wind_legend = mpatches.Patch(color='purple', label='Wind Data')
    ax.legend(handles=[temp_legend, dew_legend, alt_legend, wind_legend],
             loc='upper right', fontsize=12)

    wwa_text = "### Active Weather Watches/Advisories in Georgia:\n"
    if alerts:
        for alert in alerts:
            wwa_text += f"**{alert['severity']}**: {alert['headline']}\n"
    else:
        wwa_text += "No active alerts at this time."
    fig.text(0.5, 0.01, wwa_text, ha="center", va="bottom", fontsize=8,
            family="monospace", transform=fig.transFigure)

    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.05, right=0.95)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    await ctx.send(file=discord.File(buf, filename="ga_detailed_weather.png"))
    plt.close()
