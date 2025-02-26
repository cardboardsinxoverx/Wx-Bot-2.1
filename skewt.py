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

# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent

# Create the bot object
bot = commands.Bot(command_prefix='$', intents=intents)

# Station coordinates (latitude, longitude) for forecast data (kept for reference)
STATIONS = {
    'FFC': (33.36, -84.57),   # Peachtree City, GA
    'LIX': (30.34, -89.82),   # Slidell, LA
    'OUN': (35.18, -97.44),   # Norman, OK
    'KSLE': (44.92, -123.0),  # Salem/McNary, OR
    # Add more stations as needed
}

# Helper functions for forecast data (commented out since forecast isn’t working)
"""
def get_latest_gfs_ncss():
    # Fetch the latest GFS dataset and NCSS object.
    catalog = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml')
    latest_dataset = catalog.latest
    ncss = NCSS(latest_dataset.access_urls['NetcdfServer'])
    return ncss, latest_dataset

def fetch_forecast_data(ncss, valid_time, lat, lon):
    # Fetch GFS forecast data for a specific point and time.
    query = ncss.query()
    query.lonlat_point(lon, lat)
    query.time(valid_time)
    query.variables('Temperature_isobaric', 'Relative_humidity_isobaric',
                    'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric',
                    'Geopotential_height_isobaric')
    query.accept('netcdf4')
    data = ncss.get_data(query)
    p = data.variables['isobaric'][:] * units.hPa
    T = data.variables['Temperature_isobaric'][:] * units.kelvin
    rh = data.variables['Relative_humidity_isobaric'][:] * units.percent
    u = data.variables['u-component_of_wind_isobaric'][:] * units('m/s')
    v = data.variables['v-component_of_wind_isobaric'][:] * units('m/s')
    z = data.variables['Geopotential_height_isobaric'][:] * units.meter
    Td = mpcalc.dewpoint_from_relative_humidity(T, rh)
    df = pd.DataFrame({
        'pressure': p.magnitude,
        'height': z.magnitude,
        'temperature': T.to('degC').magnitude,
        'dewpoint': Td.to('degC').magnitude,
        'u': u.magnitude,
        'v': v.magnitude
    })
    return df
"""

# Define temperature advection and total totals functions
def calculate_temperature_advection(p, u, v, lat):
    """
    Calculate temperature advection using thermal wind approximation.
    """
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

def calculate_total_totals(p, T, Td):
    """
    Calculate the Total Totals Index (TT-INDEX).
    """
    idx_850 = np.argmin(np.abs(p - 850 * units.hPa))
    idx_500 = np.argmin(np.abs(p - 500 * units.hPa))
    T_850 = T[idx_850].to('degC')
    Td_850 = Td[idx_850].to('degC')
    T_500 = T[idx_500].to('degC')
    TT = (T_850.magnitude + Td_850.magnitude) - 2 * T_500.magnitude
    return TT * units.dimensionless

@bot.command()
async def skewt(ctx, *args):
    """
    Fetches observed sounding data and generates an enhanced Skew-T diagram.

    Usage:
    - Observed: $skewt <station> <time> (e.g., $skewt FFC 12Z)

    Parameters:
    - ctx: Discord context object
    - args: Variable arguments for station code and time
    """
    try:
        if len(args) == 2:
            # Observed sounding
            station_code, sounding_time = args
            station_code = station_code.upper()
            sounding_time = sounding_time.upper()
            if sounding_time not in ['00Z', '12Z']:
                raise ValueError("Invalid sounding time. Use '00Z' or '12Z'.")
            utc_time = datetime.datetime.now(pytz.UTC)
            year, month, day = utc_time.year, utc_time.month, utc_time.day
            if sounding_time == "12Z":
                hour = 12
            elif sounding_time == "00Z":
                if utc_time.hour >= 12:
                    day += 1
                hour = 0
            now = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=pytz.UTC)
            df = WyomingUpperAir.request_data(now, station_code)
            if df is None or df.empty:
                raise ValueError(f"No observed data for {station_code} at {sounding_time}.")
            station_lat = df['latitude'][0]
            station_lon = df['longitude'][0]
            title = f"{station_code} - {now.strftime('%Y-%m-%d %HZ')}"
            data_type = "observed"
        else:
            raise ValueError("Invalid arguments. Use `$skewt <station> <time>` for observed soundings. Forecast is disabled for now.")

        # Forecast sounding block (commented out)
        """
        elif len(args) == 3:
            # Forecast sounding
            station_code, model, forecast_hour = args
            station_code = station_code.upper()
            model = model.lower()
            forecast_hour = int(forecast_hour)
            if model != 'gfs':
                raise ValueError("Only 'gfs' model is supported currently.")
            if station_code not in STATIONS:
                raise ValueError(f"Station {station_code} not found in database.")
            lat, lon = STATIONS[station_code]
            utc_time = datetime.datetime.now(pytz.UTC)
            run_hour = (utc_time.hour // 6) * 6  # Align with GFS run times (00, 06, 12, 18)
            run_time = datetime.datetime(utc_time.year, utc_time.month, utc_time.day, run_hour, 0, 0, tzinfo=pytz.UTC)
            valid_time = run_time + datetime.timedelta(hours=forecast_hour)
            ncss, latest_dataset = get_latest_gfs_ncss()
            df = fetch_forecast_data(ncss, valid_time, lat, lon)
            if df is None or df.empty:
                raise ValueError(f"No forecast data for {station_code} with {model} at {forecast_hour}-hour lead.")
            station_lat = lat
            station_lon = lon
            title = f"{station_code} {model.upper()} Forecast {forecast_hour}-hr ({valid_time.strftime('%Y-%m-%d %HZ')})"
            data_type = "forecast"
        """

        # Data Processing
        df.drop_duplicates(subset=['pressure'], keep='first', inplace=True)
        z = df['height'].values * units.m
        p = df['pressure'].values * units.hPa

        # Corrected check for wind components
        if 'u' in df.columns and 'v' in df.columns:
            # This won’t run since forecast is disabled, but kept for future use
            T = df['temperature'].values * units.degC
            Td = df['dewpoint'].values * units.degC
            u = df['u'].values * units('m/s')
            v = df['v'].values * units('m/s')
        else:
            # Observed data
            T = df['temperature'].values * units.degC
            Td = df['dewpoint'].values * units.degC
            wind_speed = df['speed'].values * units.knots
            wind_dir = df['direction'].values * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
            u = u.to('m/s')
            v = v.to('m/s')

        # Sort by decreasing pressure
        sort_indices = np.argsort(-p)
        z, p, T, Td, u, v = z[sort_indices], p[sort_indices], T[sort_indices], Td[sort_indices], u[sort_indices], v[sort_indices]
        T_kelvin = T.to('kelvin')
        Td_kelvin = Td.to('kelvin')

        # Calculations (LCL, parcels, indices)
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

        new_p = np.append(p[p > lcl_pressure], lcl_pressure)
        new_t = np.append(T[p > lcl_pressure], lcl_temperature.to('degC'))
        lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

        (u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, z)
        *_, total_helicity1 = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km, storm_u=u_storm, storm_v=v_storm)
        *_, total_helicity3 = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km, storm_u=u_storm, storm_v=v_storm)
        *_, total_helicity6 = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km, storm_u=u_storm, storm_v=v_storm)

        ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
        bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
        ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
        bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
        ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
        bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

        sig_tor = mpcalc.significant_tornado(sbcape, lcl_height, total_helicity3, bshear3).to_base_units()
        super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

        # Plotting (using your original layout)
        fig = plt.figure(figsize=(30, 15))
        fig.set_facecolor('lightsteelblue')
        skew = SkewT(fig, rotation=45, rect=(0.03, 0.05, 0.55, 0.92))

        x1 = np.linspace(-100, 40, 8)
        x2 = np.linspace(-90, 50, 8)
        y = [1050, 100]
        for i in range(0, 8):
            skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='#ebbe2a', alpha=0.25, zorder=1)

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
        skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black', label='LCL')
        skew.shade_cin(p, T, prof, Td)
        skew.shade_cape(p, T, prof)

        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
        skew.ax.set_xlabel('Temperature (°C)', weight='bold')
        skew.ax.set_ylabel(f'Pressure ({p.units:~P})', weight='bold')
        skew.plot_dry_adiabats(linewidth=1.5, color='brown', label='Dry Adiabat')
        skew.plot_moist_adiabats(linewidth=1.5, color='purple', label='Moist Adiabat')
        skew.plot_mixing_lines(linewidth=1.5, color='lime', label='Mixing Ratio (g/kg)')

        # Temperature Advection Plot
        adv_ax = plt.axes([0.59, 0.05, 0.08, 0.90])
        temp_advection = temperature_advection.magnitude
        max_adv = 5
        norm = Normalize(vmin=-max_adv, vmax=max_adv)
        colors = plt.cm.coolwarm(norm(temp_advection))
        adv_ax.barh(p.magnitude, temp_advection, color=colors, height=8, align='center')
        adv_ax.set_ylim(1000, 100)
        adv_ax.set_xlim(-max_adv, max_adv)
        adv_ax.set_xlabel('Temp. Adv. (°C/hr)')
        adv_ax.set_ylabel('Pressure (hPa)')
        adv_ax.axvline(0, color='black', linewidth=0.5)
        adv_ax.set_facecolor('#fafad2')
        adv_ax.text(0.05, 0.95, f'STP: {sig_tor[0]:.2f}', transform=adv_ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        sm = ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=adv_ax, orientation='vertical', pad=0.05)
        cbar.set_label('Temp. Adv. (°C/hr)')

        # Hodograph
        ax_hodo = plt.axes((0.61, 0.32, 0.4, 0.4))
        h = Hodograph(ax_hodo, component_range=60.)
        h.add_grid(increment=20, linestyle='--', linewidth=1)
        h.plot(u, v)
        ax_hodo.set_xlabel('U component (m/s)')
        ax_hodo.set_ylabel('V component (m/s)')
        ax_hodo.set_title(f'Hodograph {station_code} {now.strftime("%Y-%m-%d %HZ")}')
        cmap = LinearSegmentedColormap.from_list('my_cmap', ['purple', 'blue', 'green', 'yellow', 'orange', 'red'])
        norm = Normalize(vmin=z.min().magnitude, vmax=z.max().magnitude)
        colored_line = h.plot_colormapped(u, v, c=z.magnitude, linewidth=6, cmap=cmap, label='0-12km WIND')
        cbar = plt.colorbar(colored_line, ax=h.ax, orientation='vertical', pad=0.01)
        cbar.set_label('Height (m)')

        RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
        h.ax.text((RM[0].magnitude + 0.5), (RM[1].magnitude - 0.5), 'RM', weight='bold', ha='left', fontsize=13, alpha=0.6)
        h.ax.text((LM[0].magnitude + 0.5), (LM[1].magnitude - 0.5), 'LM', weight='bold', ha='left', fontsize=13, alpha=0.6)
        h.ax.text((MW[0].magnitude + 0.5), (MW[1].magnitude - 0.5), 'MW', weight='bold', ha='left', fontsize=13, alpha=0.6)
        h.ax.arrow(0, 0, RM[0].magnitude - 0.3, RM[1].magnitude - 0.3, linewidth=2, color='red',
                   alpha=0.5, label='Bunkers RM Vector', length_includes_head=True, head_width=2)
        u_rm, v_rm = RM
        h.ax.fill_between([0, u_rm.magnitude], [0, v_rm.magnitude], color='lightpink', alpha=0.3, label='RM Vector Area')

        # Indices (exact positions from your code)
        plt.figtext(0.71, 0.15, 'SBCAPE: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.15, f'{sbcape:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.13, 'SBCIN: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.13, f'{sbcin:.0f~P}', weight='bold', fontsize=12, color='purple', ha='right')
        plt.figtext(0.71, 0.11, 'MLCAPE: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.11, f'{mlcape:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.09, 'MLCIN: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.09, f'{mlcin:.0f~P}', weight='bold', fontsize=12, color='purple', ha='right')
        plt.figtext(0.71, 0.07, 'MUCAPE: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.07, f'{mucape:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.05, 'MUCIN: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.05, f'{mucin:.0f~P}', weight='bold', fontsize=12, color='purple', ha='right')
        plt.figtext(0.71, 0.03, 'TT-INDEX: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.03, f'{total_totals:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.01, 'K-INDEX: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.01, f'{kindex:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')

        plt.figtext(0.88, 0.15, '0-1km SRH: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.15, f'{total_helicity1:.0f~P}', weight='bold', fontsize=12, color='navy', ha='right')
        plt.figtext(0.88, 0.13, '0-1km SHEAR: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.13, f'{bshear1:.0f~P}', weight='bold', fontsize=12, color='blue', ha='right')
        plt.figtext(0.88, 0.11, '0-3km SRH: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.11, f'{total_helicity3:.0f~P}', weight='bold', fontsize=12, color='navy', ha='right')
        plt.figtext(0.88, 0.09, '0-3km SHEAR: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.09, f'{bshear3:.0f~P}', weight='bold', fontsize=12, color='blue', ha='right')
        plt.figtext(0.88, 0.07, '0-6km SRH: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.07, f'{total_helicity6:.0f~P}', weight='bold', fontsize=12, color='navy', ha='right')
        plt.figtext(0.88, 0.05, '0-6km SHEAR: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.05, f'{bshear6:.0f~P}', weight='bold', fontsize=12, color='blue', ha='right')
        plt.figtext(0.88, 0.03, 'SIG TORNADO: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.03, f'{sig_tor[0]:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')
        plt.figtext(0.88, 0.01, 'SUPERCELL COMP: ', weight='bold', fontsize=12, color='black', ha='left')
        plt.figtext(0.96, 0.01, f'{super_comp[0]:.0f~P}', weight='bold', fontsize=12, color='red', ha='right')

        # Signature and Logos (exact placement from your code)
        plt.figtext(0.71, 0.27, f"Plot Created With MetPy (C) Evan J Lane 2024\nData Source: University of Wyoming\nImage Created: " +
                    utc_time.strftime('%H:%M UTC'), fontsize=16, fontweight='bold', verticalalignment='top',
                    bbox=dict(facecolor='#ebbe2a', alpha=0.7, edgecolor='none', pad=3))

        add_metpy_logo(fig, 85, 85, size='small')

        logo_img = plt.imread('/home/evanl/Documents/bot/boxlogo2.png')  # Update this path if needed
        imagebox = OffsetImage(logo_img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (1.10, 1.20), xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
        ax_hodo.add_artist(ab)

        usmc_img = plt.imread('/home/evanl/Documents/bot/Georgia_Bulldogs_logo.png')  # Update this path if needed
        imagebox = OffsetImage(usmc_img, zoom=0.97)
        abx = AnnotationBbox(imagebox, (0.45, 1.20), xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
        ax_hodo.add_artist(abx)

        skew.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Skew-T Legend', prop={'size': 10})
        h.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Hodograph Legend', prop={'size': 10})

        skew.ax.set_facecolor('#d9deda')
        plt.suptitle('UPPER AIR SOUNDING', fontsize=24, fontweight='bold')
        plt.title(title, weight='bold', size=20, color='black')
        skew.ax.set_title(f'Skew-T Log-P Diagram - Latitude: {station_lat:.2f}, Longitude: {station_lon:.2f}')

        # Save and send
        temp_image_path = f"skewt_{station_code}_{data_type}.png"
        plt.savefig(temp_image_path, format='png')
        plt.close(fig)
        await ctx.send(file=discord.File(temp_image_path))
        os.remove(temp_image_path)

    except requests.exceptions.RequestException as e:
        await ctx.send(f"Error fetching sounding data for {station_code}: {e}")
    except AttributeError as e:
        await ctx.send(f"Error processing sounding data for {station_code}: {e}")
    except ValueError as e:
        await ctx.send(str(e))
    except Exception as e:
        await ctx.send(f"An unexpected error occurred: {e}")

# UPDATED 20250226 @ 1450 EST - EJL
