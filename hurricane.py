import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For custom colorbar positioning
import cartopy.crs as ccrs
import discord
from discord.ext import commands
from tropycal.realtime import Realtime
import config

# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent if you need to read message content

# Create the bot object
bot = commands.Bot(command_prefix='$', intents=intents)

# Define paths to store historical model and ensemble runs
historical_models_path = '/home/evanl/Documents/historical_models.pkl'
historical_ensembles_path = '/home/evanl/Documents/historical_ensembles.pkl'

# Function to save historical model forecast data
def save_historical_model_data(models_data):
    # Load existing historical model runs if the file exists
    if os.path.exists(historical_models_path):
        with open(historical_models_path, 'rb') as f:
            historical_models = pickle.load(f)
    else:
        historical_models = []

    # Append the new model data
    historical_models.append(models_data)

    # Limit the number of stored historical runs to avoid excessive memory usage
    historical_models = historical_models[-5:]  # Keep only the last 5 runs

    # Save updated historical model runs back to the file
    with open(historical_models_path, 'wb') as f:
        pickle.dump(historical_models, f)

# Function to save historical ensemble forecast data
def save_historical_ensemble_data(ensembles_data):
    # Load existing historical ensemble runs if the file exists
    if os.path.exists(historical_ensembles_path):
        with open(historical_ensembles_path, 'rb') as f:
            historical_ensembles = pickle.load(f)
    else:
        historical_ensembles = []

    # Append the new ensemble data
    historical_ensembles.append(ensembles_data)

    # Save updated historical ensemble runs back to the file
    with open(historical_ensembles_path, 'wb') as f:
        pickle.dump(historical_ensembles, f)

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

            # Plot models (including historical model trends)
            try:
                fig, ax = plt.subplots(figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})

                # Plot current models
                storm.plot_models(ax=ax)

                # Load historical advisories for plotting
                if os.path.exists(historical_models_path):  # <-- This line was incorrect
                    with open(historical_models_path, 'rb') as f:
                        historical_advisories = pickle.load(f)
                else:
                    historical_advisories = []

                # Plot historical advisories (using lat, lon data)
                for idx, advisory in enumerate(historical_advisories):
                    try:
                        ax.plot(advisory['lon'], advisory['lat'], marker='o', linestyle='-.', alpha=0.5, label=f"Historical {idx+1}")
                    except Exception as e:
                        print(f"Error plotting historical advisory model: {e}")

                # Add a white patch to cover the Tropycal-generated title
                fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color='white', zorder=5))

                # Add custom title above the plot
                fig.suptitle(f"Model Forecast Tracks for {storm.name} with Historical Runs", fontsize=16, y=0.98)

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

            # Plot models (including historical model trends)
            try:
                fig, ax = plt.subplots(figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})

                # Plot current models and capture the data for saving
                storm.plot_models(ax=ax)

                # Capturing the current model tracks by accessing the plotted lines
                models_data = {}
                for line in ax.get_lines():
                    label = line.get_label()
                    if label not in ['NHC', 'Best Track']:
                        models_data[label] = {
                            'lon': line.get_xdata().tolist(),
                            'lat': line.get_ydata().tolist()
                        }
                # Save the latest model forecast data for future comparison
                save_historical_model_data(models_data)

                # Load historical model data for plotting
                if os.path.exists(historical_models_path):  # Use the correct variable name
                    with open(historical_models_path, 'rb') as f:  # Use the correct variable name
                        historical_models = pickle.load(f)
                else:
                    historical_models = []

                # Plot historical model runs (with alpha and dashed linestyle)
                for idx, model_run in enumerate(historical_models[:-1]):  # Exclude current run
                    try:
                        for model_name, model_coords in model_run.items():
                            ax.plot(model_coords['lon'], model_coords['lat'], linestyle='--', alpha=1.0,
                                    label=f"{model_name} (Run {idx + 1})")
                    except Exception as e:
                        print(f"Error plotting historical model run: {e}")

                # Add a white patch to cover the Tropycal-generated title
                fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color='white', zorder=5))

                # Add custom title above the plot
                fig.suptitle(f"Model Forecast Tracks for {storm.name} with Historical Runs", fontsize=16, y=0.98)

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

            # Plot ensembles (including historical ensembles)
            try:
                ensemble_data = storm.get_ensemble_forecasts()

                if ensemble_data is None or len(ensemble_data) == 0:
                    await ctx.send("No ensemble data available for this storm.")
                else:
                    # Create a Matplotlib figure and axis with a geographic projection
                    fig, ax = plt.subplots(figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})

                    # Plot current ensembles and capture the data for saving
                    storm.plot_ensembles(forecast=ensemble_data, ax=ax, title='')  # Suppress the default title

                    # Capturing the current ensemble tracks by accessing the plotted lines
                    ensembles_data = []
                    for line in ax.get_lines():
                        label = line.get_label()
                        if label != 'Best Track':
                            ensembles_data.append({
                                'lon': line.get_xdata().tolist(),
                                'lat': line.get_ydata().tolist()
                            })

                    # Add ensembles plot (similar approach)
                    try:
                        fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                        storm.plot_ensembles(ax=ax)

                        # Add a white patch to cover the Tropycal-generated title
                        fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color='white', zorder=5))

                        # Add the custom title above the plot
                        fig.suptitle(f"GEFS Forecast Tracks for {storm.name}", fontsize=16, y=0.95)

                        # Adjust the layout
                        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95)

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

                    plt.tight_layout()

                    # Adjust the colorbar if needed (check if there's a collection to base it on)
                    if hasattr(ax, 'collections') and ax.collections:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.1)
                        plt.colorbar(ax.collections[-1], cax=cax)
                    else:
                        print("No collections found in ax; skipping colorbar adjustment.")

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
