import datetime
import os
import pytz
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from matplotlib.patches import Rectangle
from metpy.plots.wx_symbols import sky_cover
from metpy.plots import add_metpy_logo
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap
from siphon.simplewebservice.wyoming import WyomingUpperAir
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import discord
from discord.ext import commands
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import asyncio
import logging
import pint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent

# Create the bot object
bot = commands.Bot(command_prefix='$', intents=intents)

# Station coordinates (latitude, longitude)
STATIONS = {
    'FFC': (33.36, -84.57),   # Peachtree City, GA
    'LIX': (30.34, -89.82),   # Slidell, LA
    'OUN': (35.18, -97.44),   # Norman, OK
    'KSLE': (44.92, -123.0),  # Salem/McNary, OR
}

async def fetch_sounding_data(ctx, time, station_code, max_retries=5, delay=5):
    """Fetch sounding data with retries for server errors."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching data for {station_code} at {time}")
            df = await asyncio.to_thread(WyomingUpperAir.request_data, time, station_code)
            if df is None or df.empty:
                raise ValueError("Received empty or None DataFrame from WyomingUpperAir")
            logger.info(f"Successfully fetched data for {station_code}: {len(df)} rows")
            return df
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 503:
                if attempt < max_retries - 1:
                    await ctx.send(f"Server busy (503). Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"503 error on attempt {attempt + 1} for {station_code}")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Max retries ({max_retries}) reached—server still busy!")
            else:
                logger.error(f"HTTP error fetching data for {station_code}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error fetching data for {station_code}: {e}")
            raise
    raise Exception("Max retries reached—server still too busy!")

def calculate_temperature_advection(p, u, v, lat):
    """Calculate temperature advection using thermal wind approximation."""
    try:
        f = mpcalc.coriolis_parameter(lat * units.degrees)
        R_d = 287 * units('J/(kg K)')
        p_pa = p.to('Pa')

        delta_p = np.diff(p_pa)
        delta_u = np.diff(u)
        delta_v = np.diff(v)
        p_avg = (p_pa[:-1] + p_pa[1:]) / 2

        dTdx_layers = - (f * p_avg / R_d) * (delta_v / delta_p)
        dTdy_layers = (f * p_avg / R_d) * (delta_u / delta_p)

        n = len(p)
        dTdx = np.zeros(n) * dTdx_layers.units
        dTdy = np.zeros(n) * dTdy_layers.units

        dTdx[0], dTdy[0] = dTdx_layers[0], dTdy_layers[0]
        dTdx[n-1], dTdy[n-1] = dTdx_layers[-1], dTdy_layers[-1]
        for k in range(1, n-1):
            dTdx[k] = (dTdx_layers[k-1] + dTdx_layers[k]) / 2
            dTdy[k] = (dTdy_layers[k-1] + dTdy_layers[k]) / 2

        advection = - (u * dTdx + v * dTdy)
        return (advection * 3600 * units('s/hour')).to('degC/hour')
    except Exception as e:
        logger.error(f"Error in temperature advection calculation: {e}")
        return np.full(len(p), np.nan) * units('degC/hour')

def calculate_total_totals(p, T, Td):
    """Calculate the Total Totals Index (TT-INDEX)."""
    try:
        idx_850 = np.argmin(np.abs(p - 850 * units.hPa))
        idx_500 = np.argmin(np.abs(p - 500 * units.hPa))
        T_850 = T[idx_850].to('degC')
        Td_850 = Td[idx_850].to('degC')
        T_500 = T[idx_500].to('degC')
        TT = (T_850.magnitude + Td_850.magnitude) - 2 * T_500.magnitude
        return TT * units.dimensionless
    except Exception as e:
        logger.error(f"Error in Total Totals calculation: {e}")
        return np.nan * units.dimensionless

@bot.command()
async def skewt(ctx, *args):
    """Generate an enhanced Skew-T diagram from observed sounding data."""
    try:
        if len(args) != 2:
            raise ValueError("Usage: `$skewt <station> <time>` (e.g., `$skewt FFC 12Z`)")

        station_code, sounding_time = args
        station_code = station_code.upper()
        sounding_time = sounding_time.upper()

        if sounding_time not in ['00Z', '12Z']:
            raise ValueError("Invalid time. Use '00Z' or '12Z'.")

        utc_time = datetime.datetime.now(pytz.UTC)
        year, month, day = utc_time.year, utc_time.month, utc_time.day
        hour = 12 if sounding_time == "12Z" else 0
        if sounding_time == "00Z" and utc_time.hour >= 12:
            day += 1
        now = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=pytz.UTC)

        # Fetch and process data
        df = await fetch_sounding_data(ctx, now, station_code)
        station_lat = df['latitude'][0]
        station_lon = df['longitude'][0]
        title = f"{station_code} - {now.strftime('%Y-%m-%d %HZ')}"
        data_type = "observed"

        # Data cleaning
        df.drop_duplicates(subset=['pressure'], keep='first', inplace=True)
        essential_cols = ['pressure', 'height', 'temperature', 'dewpoint', 'u', 'v']
        if not all(col in df.columns for col in ['u', 'v']):
            essential_cols = ['pressure', 'height', 'temperature', 'dewpoint', 'speed', 'direction']
        df = df.dropna(subset=essential_cols)

        if df.empty:
            raise ValueError(f"No valid data for {station_code} after cleaning.")

        # Extract and sort data
        z = df['height'].values * units.m
        p = df['pressure'].values * units.hPa
        T = df['temperature'].values * units.degC
        Td = df['dewpoint'].values * units.degC

        if 'u' in df.columns and 'v' in df.columns:
            u = df['u'].values * units('m/s')
            v = df['v'].values * units('m/s')
        else:
            wind_speed = df['speed'].values * units.knots
            wind_dir = df['direction'].values * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
            u = u.to('m/s')
            v = v.to('m/s')

        sort_indices = np.argsort(-p.magnitude)
        z, p, T, Td, u, v = [arr[sort_indices] for arr in [z, p, T, Td, u, v]]
        T_kelvin = T.to('kelvin')
        Td_kelvin = Td.to('kelvin')

        # Check for NaN values
        for name, arr in [('pressure', p), ('temperature', T), ('dewpoint', Td), ('u', u), ('v', v)]:
            if np.any(np.isnan(arr.magnitude)):
                logger.warning(f"NaN values detected in {name} array")

        # Atmospheric calculations
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)

        ml_depth = 300 * units.hPa
        ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=ml_depth)
        ml_prof = mpcalc.parcel_profile(p, ml_t, ml_td).to('degC')
        mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=ml_depth)

        mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=100 * units.hPa)
        mu_prof = mpcalc.parcel_profile(p, mu_t, mu_td).to('degC')
        mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=100 * units.hPa)

        theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)
        min_theta_e_idx = np.argmin(theta_e[p > 500 * units.hPa])
        down_prof = mpcalc.parcel_profile(p, T[min_theta_e_idx], Td[min_theta_e_idx]).to('degC')

        wet_bulb = mpcalc.wet_bulb_temperature(p, T_kelvin, Td_kelvin)
        kindex = mpcalc.k_index(p, T_kelvin, Td_kelvin)
        total_totals = calculate_total_totals(p, T, Td)
        temperature_advection = calculate_temperature_advection(p, u, v, station_lat)

        lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
        el_pressure, el_temperature = mpcalc.el(p, T, Td)

        # Freezing level calculation with safety
        idx = np.where(T.magnitude < 0)[0]
        p_freeze = None
        if len(idx) > 0 and idx[0] > 0:
            idx = idx[0]
            T1, T2 = T[idx-1], T[idx]
            p1, p2 = p[idx-1], p[idx]
            if T2 != T1 and not np.any(np.isnan([T1.magnitude, T2.magnitude, p1.magnitude, p2.magnitude])):
                p_freeze = p1 + (0 * units.degC - T1) * (p2 - p1) / (T2 - T1)
                if p_freeze < 100 * units.hPa or p_freeze > 1000 * units.hPa:
                    logger.warning(f"Freezing level pressure {p_freeze} out of bounds")
                    p_freeze = None

        # Calculate storm motion and helicity/shear
        RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
        storm_u, storm_v = RM  # Use right-mover storm motion for helicity

        # Calculate storm-relative helicity
        total_helicity1, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km, storm_u=storm_u, storm_v=storm_v)
        total_helicity3, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km, storm_u=storm_u, storm_v=storm_v)
        total_helicity6, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km, storm_u=storm_u, storm_v=storm_v)

        # Calculate bulk shear
        bshear1 = mpcalc.bulk_shear(p, u, v, depth=1 * units.km)
        bshear3 = mpcalc.bulk_shear(p, u, v, depth=3 * units.km)
        bshear6 = mpcalc.bulk_shear(p, u, v, depth=6 * units.km)

        # Compute shear magnitudes
        bshear1_mag = np.sqrt(bshear1[0]**2 + bshear1[1]**2) if bshear1 is not None else np.nan * units('m/s')
        bshear3_mag = np.sqrt(bshear3[0]**2 + bshear3[1]**2) if bshear3 is not None else np.nan * units('m/s')
        bshear6_mag = np.sqrt(bshear6[0]**2 + bshear6[1]**2) if bshear6 is not None else np.nan * units('m/s')

        # Calculate LCL height
        if lcl_pressure is not None:
            lcl_height = np.interp(lcl_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m
        else:
            lcl_height = np.nan * units.m

        # Calculate significant tornado parameter (STP)
        try:
            sig_tor = mpcalc.significant_tornado(sbcape, lcl_height, total_helicity1, bshear6_mag).to_base_units()
        except Exception as e:
            logger.error(f"Error calculating sig_tor: {e}")
            sig_tor = np.nan * units.dimensionless

        # Calculate supercell composite (SCP)
        try:
            super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3_mag).to_base_units()
        except Exception as e:
            logger.error(f"Error calculating super_comp: {e}")
            super_comp = np.nan * units.dimensionless

        # Create the figure with the specified size and background color
        fig = plt.figure(figsize=(30, 15))
        fig.set_facecolor('lightsteelblue')

        # Set up the Skew-T plot with a 45-degree rotation and specified rectangle
        skew = SkewT(fig, rotation=45, rect=(0.03, 0.05, 0.55, 0.92))

        # Define temperature ranges and pressure levels for shading
        x1 = np.linspace(-100, 40, 8)  # 8 temperature points from -100°C to 40°C
        x2 = np.linspace(-90, 50, 8)   # 8 temperature points from -90°C to 50°C
        y = [1050, 100]                # Pressure levels from 1050 hPa to 100 hPa

        # Add the shaded areas
        for i in range(0, 8):
            skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='#ebbe2a', alpha=0.25, zorder=1)

        # Basic plots
        skew.plot(p, T, 'r', linewidth=2, label='Temperature')
        skew.plot(p, Td, 'g', linewidth=2, label='Dewpoint')
        skew.plot(p, wet_bulb.to('degC'), 'b', linestyle='--', linewidth=2, label='Wet Bulb')
        skew.plot(p, prof, 'k', linewidth=2.5, label='SB Parcel')
        skew.plot(p, ml_prof, 'm--', linewidth=2, label='ML Parcel')
        skew.plot(p, mu_prof, 'y--', linewidth=2, label='MU Parcel')
        skew.plot(p, down_prof, 'c--', linewidth=2, label='Downdraft')
        skew.plot_barbs(p[::2], u[::2], v[::2], color='#000000')

        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        skew.shade_cin(p, T, prof, Td)
        skew.shade_cape(p, T, prof)

        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
        skew.ax.set_xlabel('Temperature (°C)', weight='bold')
        skew.ax.set_ylabel(f'Pressure ({p.units:~P})', weight='bold')
        skew.plot_dry_adiabats(linewidth=1.5, color='brown', label='Dry Adiabat')
        skew.plot_moist_adiabats(linewidth=1.5, color='purple', label='Moist Adiabat')
        skew.plot_mixing_lines(linewidth=1.5, color='lime', label='Mixing Ratio (g/kg)')

        # Plot key levels using scatter for single points
        def plot_point(skew, pressure, temperature, marker, color, label, log_message):
            if (pressure is not None and temperature is not None and
                not np.isnan(pressure.magnitude) and not np.isnan(temperature.magnitude) and
                100 <= pressure.magnitude <= 1000 and -40 <= temperature.magnitude <= 60):
                try:
                    logger.debug(f"Plotting {label}: pressure={pressure}, temperature={temperature}")
                    skew.ax.scatter(temperature, pressure, marker='o', s=50, color=color)
                    skew.ax.text(temperature.magnitude + 2, pressure.magnitude, label,
                                 fontsize=10, color='k' if label in ['LFC', 'EL'] else color)
                    logger.info(log_message)
                except Exception as e:
                    logger.error(f"Error plotting {label}: {e}")
            else:
                logger.info(f"Skipping {label}: Invalid pressure={pressure}, temperature={temperature} "
                            f"(likely due to stable conditions or no CAPE)")

        plot_point(skew, lcl_pressure, lcl_temperature, 'o', 'black', 'LCL',
                   f"LCL plotted at {lcl_pressure}, {lcl_temperature}")
        plot_point(skew, lfc_pressure, lfc_temperature, 'o', 'black', 'LFC',
                   f"LFC plotted at {lfc_pressure}, {lfc_temperature}")
        plot_point(skew, el_pressure, el_temperature, 'o', 'black', 'EL',
                   f"EL plotted at {el_pressure}, {el_temperature}")
        plot_point(skew, p_freeze, 0 * units.degC if p_freeze is not None else None, 'o', 'blue', 'FL',
                   f"Freezing Level plotted at {p_freeze}, 0°C")

        # Temperature Advection Plot
        adv_ax = plt.axes([0.59, 0.05, 0.08, 0.90])
        temp_advection = temperature_advection.magnitude
        max_adv = 5
        norm = Normalize(vmin=-max_adv, vmax=max_adv)
        colors = plt.cm.coolwarm(norm(temp_advection))
        adv_ax.barh(p.magnitude, temp_advection, color=colors, height=8, align='center')
        adv_ax.set_ylim(1000, 100)
        adv_ax.set_xlim(-max_adv, max_adv)
        adv_ax.set_xlabel('Temp. Adv. (°C/hr)', weight='bold')
        adv_ax.set_ylabel('Pressure (hPa)', weight='bold')
        adv_ax.axvline(0, color='black', linewidth=0.5)
        adv_ax.set_facecolor('#fafad2')

        # Hodograph
        ax_hodo = plt.axes((0.61, 0.32, 0.4, 0.4))
        h = Hodograph(ax_hodo, component_range=60.)
        h.add_grid(increment=20, linestyle='--', linewidth=1)
        cmap = LinearSegmentedColormap.from_list('my_cmap', ['purple', 'blue', 'green', 'yellow', 'orange', 'red'])
        norm = Normalize(vmin=z.min().magnitude, vmax=z.max().magnitude)
        colored_line = h.plot_colormapped(u, v, c=z.magnitude, linewidth=6, cmap=cmap, label='0-12km WIND')
        cbar = plt.colorbar(colored_line, ax=h.ax, orientation='vertical', pad=0.01)
        cbar.set_label('Height (m)')
        ax_hodo.set_xlabel('U component (m/s)', weight='bold')
        ax_hodo.set_ylabel('V component (m/s)', weight='bold')
        ax_hodo.set_title(f'Hodograph {title}')

        for motion, label, offset in [(RM, 'RM', (0.5, -0.5)), (LM, 'LM', (0.5, -0.5)), (MW, 'MW', (0.5, -0.5))]:
            h.ax.text(motion[0].magnitude + offset[0], motion[1].magnitude + offset[1], label,
                      weight='bold', ha='left', fontsize=13, alpha=0.6)
        h.ax.arrow(0, 0, RM[0].magnitude - 0.3, RM[1].magnitude - 0.3, linewidth=2, color='red',
                   alpha=0.5, label='Bunkers RM Vector', length_includes_head=True, head_width=2)
        h.ax.fill([0, u[0].magnitude, RM[0].magnitude], [0, v[0].magnitude, RM[1].magnitude],
                  color='red', alpha=0.3, label='RM Vector Area')

        # Indices with unit-aware formatting
        bbox_props = dict(facecolor='#fafad2', alpha=0.7, edgecolor='none', pad=3)
        indices = [
            ('SBCAPE', sbcape if 'sbcape' in locals() else np.nan, 'red', 0.15, None),
            ('SBCIN', sbcin if 'sbcin' in locals() else np.nan, 'purple', 0.13, None),
            ('MLCAPE', mlcape if 'mlcape' in locals() else np.nan, 'red', 0.11, None),
            ('MLCIN', mlcin if 'mlcin' in locals() else np.nan, 'purple', 0.09, None),
            ('MUCAPE', mucape if 'mucape' in locals() else np.nan, 'red', 0.07, None),
            ('MUCIN', mucin if 'mucin' in locals() else np.nan, 'purple', 0.05, None),
            ('TT-INDEX', total_totals if 'total_totals' in locals() else np.nan, 'red', 0.03, None),
            ('K-INDEX', kindex if 'kindex' in locals() else np.nan, 'red', 0.01, None),
            ('SIG TORNADO', sig_tor if 'sig_tor' in locals() and sig_tor else np.nan, 'red', -0.01, None),
            ('0-1km SRH', total_helicity1 if 'total_helicity1' in locals() else np.nan, 'navy', 0.15, 0.88),
            ('0-1km SHEAR', bshear1_mag if 'bshear1_mag' in locals() else np.nan, 'blue', 0.13, 0.88),
            ('0-3km SRH', total_helicity3 if 'total_helicity3' in locals() else np.nan, 'navy', 0.11, 0.88),
            ('0-3km SHEAR', bshear3_mag if 'bshear3_mag' in locals() else np.nan, 'blue', 0.09, 0.88),
            ('0-6km SRH', total_helicity6 if 'total_helicity6' in locals() else np.nan, 'navy', 0.07, 0.88),
            ('0-6km SHEAR', bshear6_mag if 'bshear6_mag' in locals() else np.nan, 'blue', 0.05, 0.88),
            ('SUPERCELL COMP', super_comp if 'super_comp' in locals() and super_comp else np.nan, 'red', -0.01, 0.88),
        ]

        for label, value, color, y, x_left in indices:
            x_left = x_left if x_left is not None else 0.71
            x_right = 0.80 if x_left == 0.71 else 0.96
            if isinstance(value, pint.Quantity):
                if np.isnan(value.magnitude):
                    value_str = 'N/A'
                else:
                    value_str = f'{value:.0f~P}'  # Keeps units, e.g., "302 J/kg"
            else:
                if np.isnan(value):
                    value_str = 'N/A'
                else:
                    value_str = f'{value:.0f}'    # For dimensionless values, e.g., "47"
            plt.figtext(x_left, y, f'{label}: ', weight='bold', fontsize=12, color='black', ha='left', bbox=bbox_props)
            plt.figtext(x_right, y, value_str, weight='bold', fontsize=12, color=color, ha='right', bbox=bbox_props)

        # Signature and Logos
        plt.figtext(0.71, 0.23, f"Plot Created With MetPy (C) Evan J Lane 2024\nData Source: University of Wyoming\nImage Created: " +
                    utc_time.strftime('%H:%M UTC'), fontsize=16, fontweight='bold', bbox=bbox_props)
        add_metpy_logo(fig, 85, 85, size='small')

        logo_paths = {
            'boxlogo': '/home/evanl/Documents/bot/boxlogo2.png',
            'bulldogs': '/home/evanl/Documents/bot/Georgia_Bulldogs_logo.png'
        }
        for name, path, pos, zoom in [('boxlogo', logo_paths['boxlogo'], (1.10, 1.20), 0.2),
                                      ('bulldogs', logo_paths['bulldogs'], (0.45, 1.20), 0.97)]:
            try:
                logo_img = plt.imread(path)
                imagebox = OffsetImage(logo_img, zoom=zoom)
                ab = AnnotationBbox(imagebox, pos, xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
                ax_hodo.add_artist(ab)
            except FileNotFoundError:
                logger.error(f"Logo file not found: {path}")
                await ctx.send(f"Warning: Logo '{name}' not found at {path}")

        skew.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Skew-T Legend', title_fontsize=10)
        h.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Hodograph Legend', title_fontsize=10)
        skew.ax.set_facecolor('#d9deda')
        plt.suptitle('UPPER AIR SOUNDING', fontsize=24, fontweight='bold')
        plt.title(title, weight='bold', size=20, color='black')
        skew.ax.set_title(f'Skew-T Log-P Diagram - Latitude: {station_lat:.2f}, Longitude: {station_lon:.2f}')

        # Save and send
        temp_image_path = f"skewt_{station_code}_{data_type}.png"
        plt.savefig(temp_image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        await ctx.send(file=discord.File(temp_image_path))
        os.remove(temp_image_path)
        logger.info(f"Skew-T diagram generated and sent for {station_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        await ctx.send(f"Error fetching data for {station_code}: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        await ctx.send(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        await ctx.send(f"An unexpected error occurred: {e}")

# Uncomment and add your bot token to run
# bot.run('YOUR_DISCORD_BOT_TOKEN')
