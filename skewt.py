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
from matplotlib.colors import Normalize
from siphon.simplewebservice.wyoming import WyomingUpperAir
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import discord
from discord.ext import commands
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
import metpy.interpolate as mpinterpolate
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from metpy.calc import lcl, parcel_profile
from matplotlib import cm


# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent if you need to read message content

# Create the bot object
bot = commands.Bot(command_prefix='$', intents=intents)

import datetime
import pytz
from siphon.simplewebservice.wyoming import WyomingUpperAir

# --- SkewT Command ---
@bot.command()
async def skewt(ctx, station_code: str, sounding_time: str = "12Z"):
    """
    Fetches 12Z or 00Z sounding data and generates a Skew-T diagram.
    """

    try:
        station_code = station_code.upper()
        sounding_time = sounding_time.upper()

        # Get the current time in UTC
        utc_time = datetime.datetime.now(pytz.UTC)

        # Get year, month, day based on UTC time
        year, month, day = utc_time.year, utc_time.month, utc_time.day

        # Determine the correct hour for the sounding time
        if sounding_time == "12Z":
            hour = 12
        elif sounding_time == "00Z":
            # If the current UTC time is after 12Z, get today's 00Z
            if utc_time.hour >= 12:
                day += 1  # Move forward to the next day's 00Z
            hour = 0
        else:
            raise ValueError("Invalid sounding time. Please choose either '12Z' or '00Z'.")

        # Create the datetime object for the request
        now = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=pytz.UTC)

        # Fetch sounding data using Siphon
        df = WyomingUpperAir.request_data(now, station_code)

        # Check if data was fetched successfully
        if df is None:
            raise ValueError(f"No sounding data found for station {station_code} at {sounding_time}. Please check the station code and time, or try a different station.")

        print(df.head())  # Inspect the first few rows
        print(df.info())   # Print column names and data types
        print(df.isnull().sum())  # Count null values in each column

        ###############################################################
        # CALCULATE INDICES #
        ###############################################################

        try:
            # Drop rows with duplicate pressure values (in-place)
            df.drop_duplicates(subset=['pressure'], keep='first', inplace=True)

            # Convert sounding data to MetPy units
            z = df['height'].values * units.m
            p = df['pressure'].values * units.hPa
            # Extract relative humidity from the 5th column (index 4)
            rh = df.iloc[:, 4].values * units.percent
            T = df['temperature'].values * units.degC
            Td = df['dewpoint'].values * units.degC
            wind_speed = df['speed'].values * units.knots
            wind_dir = df['direction'].values * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
            # grab lat/long
            station_lat = df['latitude'][0]  # Get the latitude of the first row (assuming it's the station's location)
            station_lon = df['longitude'][0] # Get the longitude of the first row

            # Sort data by height (descending order since z represents pressure in millibars)
            sort_indices = np.argsort(-p)
            z = z[sort_indices]
            p = p[sort_indices]
            T = T[sort_indices]
            Td = Td[sort_indices]
            u = u[sort_indices]
            v = v[sort_indices]

            # Convert temperature and dewpoint to Kelvin for calculations
            T_kelvin = T.to('kelvin')
            Td_kelvin = Td.to('kelvin')

            # Calculate LCL
            lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

            # Calculate full parcel profile
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')

            # temp advection function
            def calculate_temperature_advection(p, T, u, v, station_lat, station_lon):
                """
                Calculate temperature advection from upper air sounding data.

                Parameters:
                - pressure_levels: Array of pressure levels (Pa or hPa)
                - temperatures: Array of temperatures at the pressure levels (degrees Celsius)
                - wind_u: Array of u-component of wind (east-west) at the pressure levels (m/s)
                - wind_v: Array of v-component of wind (north-south) at the pressure levels (m/s)
                - latitudes: Latitude of the sounding (degrees)
                - longitudes: Longitude of the sounding (degrees)

                Returns:
                - Temperature advection (C/hour)
                """

                # Ensure inputs are 1D arrays
                T = np.array(T)
                u = np.array(u)
                v = np.array(v)
                p = np.array(p)

                dT_dp = np.gradient(T, p)  # Gradient with respect to pressure

                # Compute the advection components
                advection_u = -u * dT_dp
                advection_v = -v * dT_dp

                # Sum components for total advection
                temperature_advection = advection_u + advection_v

                # Convert to C/hour (assuming wind is in m/s and gradient in C/hPa)
                temperature_advection *= 3600  # Convert from per second to per hour

                return temperature_advection

            # Calculate wet-bulb temperature
            wet_bulb = mpcalc.wet_bulb_temperature(p, T_kelvin, Td_kelvin)

            # Calculate indices
            kindex = mpcalc.k_index(p, T_kelvin, Td_kelvin)
            total_totals = mpcalc.total_totals_index(p, T_kelvin, Td_kelvin)

            # Mixed layer parcel properties
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=300 * units.hPa)
            ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=300 * units.hPa)
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=300 * units.hPa)

            # Most unstable parcel properties
            mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=100 * units.hPa)
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=100 * units.hPa)

            # Estimate height of LCL in meters
            new_p = np.append(p[p > lcl_pressure], lcl_pressure)
            new_t = np.append(T[p > lcl_pressure], lcl_temperature.to('degC'))  # Convert lcl_temperature to Celsius

            lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

            # Compute Surface-based CAPE
            sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)

            # Compute SRH and Bulk Shear
            (u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, z)
            *_, total_helicity1 = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km,
                                                                storm_u=u_storm, storm_v=v_storm)
            *_, total_helicity3 = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km,
                                                                storm_u=u_storm, storm_v=v_storm)
            *_, total_helicity6 = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km,
                                                                storm_u=u_storm, storm_v=v_storm)

            ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
            bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
            ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
            bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
            ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
            bshear6 = mpcalc.wind_speed(ubshr6, vbshr3)

            # Calculate Significant Tornado parameter & Supercell Composite
            sig_tor = mpcalc.significant_tornado(sbcape, lcl_height, total_helicity3, bshear3).to_base_units()
            super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

             # Convert back to degrees Celsius for CAPE/CIN calculations and plotting
            T_celsius = T_kelvin.to('degC')
            Td_celsius = Td_kelvin.to('degC')

        except Exception as e:
            await ctx.send(f"An unexpected error occurred while processing the sounding data for {station_code}: {e}")
            return


        # Calculate LCL
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

        # Calculate parcel profile
        parcel_profile = mpcalc.parcel_profile(p, T[0], Td[0])

        # Create the Skew-T plot
        fig = plt.figure(figsize=(25, 15))
        # Change the background color of the entire figure
        fig.set_facecolor('#756599')
        skew = SkewT(fig, rotation=45, rect=(0.03, 0.05, 0.70, 0.92))

        # Shade every other section between isotherms
        x1 = np.linspace(-100, 40, 8)  # The starting x values for the shaded regions
        x2 = np.linspace(-90, 50, 8)  # The ending x values for the shaded regions
        y = [1050, 100]  # The range of y values that the shades regions should cover
        for i in range(0, 8):
            skew.shade_area(y=y,
                            x1=x1[i],
                            x2=x2[i],
                            color='#ebbe2a',
                            alpha=0.25,
                            zorder=1)

        # Plot temperature, dewpoint, and wet-bulb temperature
        skew.plot(p, T, 'r', linewidth=2)  # Set linewidth to 2
        skew.plot(p, Td, 'g', linewidth=2)
        if wet_bulb is not None:
            skew.plot(p, wet_bulb.to('degC'), 'b', linestyle='--', linewidth=2)

        # Plot wind barbs at the observed pressure levels
        skew.plot_barbs(p[::2], u[::2], v[::2], color='#000000')  # Adjust [::2] to control the density of barbs

        # Set plot limits
        skew.ax.set_ylim(1000, 6)
        skew.ax.set_xlim(-40, 60)

        # Calculate LCL and plot it
        skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')


        # Plot LFC, EL
        '''if lfc_pressure != "LFC not found":
            lfc_index = np.where(p == lfc_pressure)[0][0]  # Find the index of LFC pressure
            skew.ax.axhline(y=lfc_pressure, color='#54300d', linestyle='--', label='LFC')
        if el_pressure != "EL not found":
            el_index = np.where(p == el_pressure)[0][0]  # Find the index of EL pressure
            skew.ax.axhline(y=el_pressure, color='#2f0d54', linestyle='--', label='EL')

        # Display LFC, EL on the plot if they were found
        if lfc_pressure != "LFC not found":
            plt.figtext(0.71, 0.21, f"LFC: {lfc_pressure.magnitude:.0f} hPa", fontsize=12, ha='left', color='#54300d')
        if el_pressure != "EL not found":
            plt.figtext(0.71, 0.19, f"EL: {el_pressure.magnitude:.0f} hPa", fontsize=12, ha='left', color='#2f0d54')'''



        # Plot the parcel profile only if it was calculated successfully
        if prof is not None:
            skew.plot(p, prof, 'k', linewidth=2.5)

        # Shade areas of CAPE and CIN only if prof is available
        if prof is not None:
            skew.shade_cin(p, T, prof, Td)
            skew.shade_cape(p, T, prof)

        # An example of a slanted line at constant T -- in this case the 0 isotherm
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

        # Change to adjust data limits
        skew.ax.set_adjustable('datalim')
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-20, 30)

        # Set some better labels than the default to increase readability
        skew.ax.set_xlabel(str.upper('Temperature (C)'), weight='bold')
        skew.ax.set_ylabel(str.upper(f'Pressure ({p.units:~P})'), weight='bold')

        # Set the facecolor of the skew-t object and the figure to pretty colors
        fig.set_facecolor('lightsteelblue')

        skew.ax.axvline(0, linestyle='--', color='blue', alpha=0.3)

        # Add the relevant special lines
        skew.plot_dry_adiabats(linewidth=1.5, color='brown')
        skew.plot_moist_adiabats(linewidth=1.5, color='purple')

        # Add the relevant special lines
        dry_adiabats = skew.plot_dry_adiabats(linewidth=1.5, color='brown', label='Dry Adiabat')
        moist_adiabats = skew.plot_moist_adiabats(linewidth=1.5, color='purple', label='Moist Adiabat')
        mixing_lines = skew.plot_mixing_lines(linewidth=1.5, color='lime', label='Mixing Ratio (g/kg)')

        #####################################################
        # This is for labeling the mixing ratio lines by g/kg, it doesn't throw
        # an error, but it doesn't make anything show up either
        #####################################################
        #####################################################

        # Check if mixing_lines is a LineCollection
        if isinstance(mixing_lines, LineCollection):
            # Extract individual lines from the LineCollection
            lines = mixing_lines.get_segments()
            labels = mixing_lines.get_label()  # Assuming labels are set correctly

            # Iterate through each line segment and its label
            for i, (line, label) in enumerate(zip(lines, labels)):
                # Extract the label (in g/kg) from the label
                mr_label = label

                if mr_label:
                    try:
                        # Convert the label to a float and attach units (g/kg)
                        mr = units.Quantity(float(mr_label), 'g/kg')

                        # Find the midpoint of the line segment
                        mid_x = (line[0][0] + line[1][0]) / 2
                        mid_y = (line[0][1] + line[1][1]) / 2

                        # Place the text label at the midpoint of the mixing ratio line
                        skew.plot_mixing_lines(mid_x, mid_y, f'{mr:~P}', fontsize=10, color='black', ha='center', va='center')

                    except ValueError:
                        # Handle any potential errors in case the label isn't convertible to float
                        print(f"Unable to process mixing ratio label: {mr_label}")

        else:
            # If mixing_lines is not a LineCollection, handle it as before
            for mixing_line in mixing_lines:
                # Extract the label (in g/kg) from the line (assuming labels are set correctly)
                mr_label = mixing_line.get_label()

                if mr_label:
                    try:
                        # Convert the label to a float and attach units (g/kg)
                        mr = units.Quantity(float(mr_label), 'g/kg')

                        # Find the midpoint of the line segment
                        x_data = mixing_line.get_xdata()
                        y_data = mixing_line.get_ydata()
                        mid_x = (x_data[0] + x_data[-1]) / 2
                        mid_y = (y_data[0] + y_data[-1]) / 2

                        # Place the text label at the midpoint of the mixing ratio line
                        skew.plot_mixing_lines(mid_x, mid_y, f'{mr:~P}', fontsize=10, color='black', ha='center', va='center')

                    except ValueError:
                        # Handle any potential errors in case the label isn't convertible to float
                        print(f"Unable to process mixing ratio label: {mr_label}")

        # Plot lines with labels
        skew.plot(p, T, 'r', label='Temperature')
        skew.plot(p, Td, 'g', label='Dewpoint')


        temperature_advection = calculate_temperature_advection(
            p.to('Pa').magnitude,
            T.magnitude,
            u.to('m/s').magnitude,
            v.to('m/s').magnitude,
            station_lat,   # Use your actual latitude array
            station_lon   # Use your actual longitude array
        )

        # Calculate the absolute maximum advection value for symmetric normalization
        abs_max_advection = max(abs(temperature_advection.min()), abs(temperature_advection.max()))

        # Create a custom colormap with red for negative and blue for positive values, respecting height order
        colors = ['red' if temperature_advection[i] <= 0 else 'blue' for i in range(len(p))]  # Iterate over pressure levels
        cmap_flag = LinearSegmentedColormap.from_list('flag', colors, N=len(colors))

        # Normalize temperature advection data
        norm = plt.Normalize(-abs_max_advection, abs_max_advection)

        # Create a ScalarMappable object for the colorbar
        sm = cm.ScalarMappable(cmap=cmap_flag, norm=norm)
        sm.set_array([])

        # Add the colorbar to the figure
        cbar = plt.colorbar(sm, ax=skew.ax, orientation='vertical', pad=0.02, ticks=[])
        cbar.set_label('Temperature Advection (C/hour)')

        skew.ax.legend()

        # Create a hodograph
        ax = plt.axes((0.61, 0.32, 0.4, 0.4))
        h = Hodograph(ax, component_range=60.)
        h.add_grid(increment=20, linestyle='--', linewidth=1)
        h.plot(u, v)
        #h.set_facecolor('#ab867e')


        # Add labels and title (using your preferred style from suggestions)
        ax.set_xlabel('U component (m/s)')
        ax.set_ylabel('V component (m/s)')
        ax.set_title('Hodograph of Wind Components')

        # Define the colors for the colormap
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
        cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

        # Normalize height values (z) to the range [0, 1] for color-mapping
        norm = Normalize(vmin=z.min().magnitude, vmax=z.max().magnitude)

        # Plot the hodograph with color-mapped line based on height
        colored_line = h.plot_colormapped(u, v, c=z.magnitude, linewidth=6, cmap=cmap, label='0-12km WIND')

        # Add the colorbar based on the height (z) values
        cbar = plt.colorbar(colored_line, ax=h.ax, orientation='vertical', pad=0.01)
        cbar.set_label('Height (m)')

        # Display hodograph
        plt.show()

        # Compute Bunkers storm motion and annotate it
        RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
        h.ax.text((RM[0].magnitude + 0.5), (RM[1].magnitude - 0.5), 'RM', weight='bold', ha='left',
                fontsize=13, alpha=0.6)
        h.ax.text((LM[0].magnitude + 0.5), (LM[1].magnitude - 0.5), 'LM', weight='bold', ha='left',
                fontsize=13, alpha=0.6)
        h.ax.text((MW[0].magnitude + 0.5), (MW[1].magnitude - 0.5), 'MW', weight='bold', ha='left',
                fontsize=13, alpha=0.6)

        # Add Bunkers RM vector arrow to the plot
        h.ax.arrow(0, 0, RM[0].magnitude - 0.3, RM[1].magnitude - 0.3, linewidth=2, color='red',
                alpha=0.5, label='Bunkers RM Vector', length_includes_head=True, head_width=2)

        #################################################################################
        # INDICES ANALYSIS
        #################################################################################

        # There is a lot we can do with this data operationally, so let's plot some of
        # these values right on the plot, in the box we made
        # First lets plot some thermodynamic parameters
        plt.figtext(0.71, 0.15, 'SBCAPE: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.15, f'{sbcape:.0f~P}', weight='bold',
                    fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.13, 'SBCIN: ', weight='bold',
                    fontsize=12, color='black', ha='left')
        plt.figtext(0.80, 0.13, f'{sbcin:.0f~P}', weight='bold',
                    fontsize=12, color='purple', ha='right')
        plt.figtext(0.71, 0.11, 'MLCAPE: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.11, f'{mlcape:.0f~P}', weight='bold',
                    fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.09, 'MLCIN: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.09, f'{mlcin:.0f~P}', weight='bold',
                    fontsize=12, color='purple', ha='right')
        plt.figtext(0.71, 0.07, 'MUCAPE: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.07, f'{mucape:.0f~P}', weight='bold',
                    fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.05, 'MUCIN: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.05, f'{mucin:.0f~P}', weight='bold',
                    fontsize=12, color='purple', ha='right')
        plt.figtext(0.71, 0.03, 'TT-INDEX: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.03, f'{total_totals:.0f~P}', weight='bold',
                    fontsize=12, color='red', ha='right')
        plt.figtext(0.71, 0.01, 'K-INDEX: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.80, 0.01, f'{kindex:.0f~P}', weight='bold',
                    fontsize=12, color='red', ha='right')

        # now some kinematic parameters
        plt.figtext(0.88, 0.15, '0-1km SRH: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.15, f'{total_helicity1:.0f~P}',
                    weight='bold', fontsize=12, color='navy', ha='right')
        plt.figtext(0.88, 0.13, '0-1km SHEAR: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.13, f'{bshear1:.0f~P}', weight='bold',
                    fontsize=12, color='blue', ha='right')
        plt.figtext(0.88, 0.11, '0-3km SRH: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.11, f'{total_helicity3:.0f~P}',
                    weight='bold', fontsize=12, color='navy', ha='right')
        plt.figtext(0.88, 0.09, '0-3km SHEAR: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.09, f'{bshear3:.0f~P}', weight='bold',
                    fontsize=12, color='blue', ha='right')
        plt.figtext(0.88, 0.07, '0-6km SRH: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.07, f'{total_helicity6:.0f~P}',
                    weight='bold', fontsize=12, color='navy', ha='right')
        plt.figtext(0.88, 0.05, '0-6km SHEAR: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.05, f'{bshear6:.0f~P}', weight='bold',
                    fontsize=12, color='blue', ha='right')
        plt.figtext(0.88, 0.03, 'SIG TORNADO: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.03, f'{sig_tor[0]:.0f~P}', weight='bold', fontsize=12,
                    color='red', ha='right')
        plt.figtext(0.88, 0.01, 'SUPERCELL COMP: ', weight='bold', fontsize=12,
                    color='black', ha='left')
        plt.figtext(0.96, 0.01, f'{super_comp[0]:.0f~P}', weight='bold', fontsize=12,
                    color='red', ha='right')

        # Add temperature advection to the plot
       # plt.figtext(0.88, 0.17, 'MEAN T ADV: ', weight='bold', fontsize=12, color='black', ha='left')
        #plt.figtext(0.96, 0.17, f'{temperature_advection:.2f} \u00B0C/hr', weight='bold', fontsize=12, color='red', ha='right')



        # Add signature
        plt.figtext(0.71, 0.27, # place above indices
                    "Plot Created With MetPy (C) Evan J Lane 2024\nData Source: Univeristy of Wyoming\nImage Created: " +
                    utc_time.strftime('%H:%M UTC'),
                    fontsize=16,
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(facecolor='#ebbe2a', alpha=0.7, edgecolor='none', pad=3))

        ##########################################################################
        # LOGOS AND SEND PLOT
        ##########################################################################

        # Add MetPy logo
        add_metpy_logo(fig, 85, 85, size='small')

         # METOC Logo
        logo_img = plt.imread('/home/evanl/Documents/boxlogo2.png')  # Replace with your logo's actual path
        imagebox = OffsetImage(logo_img, zoom=0.2)  # Adjust 'zoom' to control logo size
        ab = AnnotationBbox(imagebox, (1.10, 1.20), xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
        ax.add_artist(ab)

        usmc_img = plt.imread('/home/evanl/Documents/uga_logo.png')  # Replace with your logo's actual path
        imagebox = OffsetImage(usmc_img, zoom=0.1)  # Adjust 'zoom' to control logo size
        abx = AnnotationBbox(imagebox, (0.45, 1.20), xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
        ax.add_artist(abx)

        # OG legend
        skewleg = skew.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Skew-T Legend', prop={'size':10})
        # OG hodo legend, improved
        hodoleg = h.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Hodograph Legend', prop={'size':10})

        skew.ax.set_facecolor('#d9deda')


        # Add labels and title
        plt.title(f'{station_code} {now.strftime("%Y-%m-%d %HZ")}', weight='bold', size=20, color='black')


        # Replace 'your_data' with the actual variable holding your dataset
        lat = station_lon
        lon = station_lat

        # Set the title with latitude and longitude
        skew.ax.set_title(f'Skew-T Log-P Diagram - Latitude: {lat:.2f}, Longitude: {lon:.2f}')

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
