# Python script to monitor interannual variability for station data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from .preparation_data import (
        apply_scientific_plot_style,
        set_scientific_title,
        style_scientific_grid,
    )
except ImportError:
    from preparation_data import (
        apply_scientific_plot_style,
        set_scientific_title,
        style_scientific_grid,
    )

## Translations: 
TRANSLATIONS = {
    "fr": {
        "title": "Précipitations cumulées",
        "title_temp": "Températures moyennes",
        "reference": "Réf.",
        "anomaly": "Anomalie",
        "anomaly_axis": "Anomalie (mm)",
        "anomaly_axis_temp": "Anomalie (°C)",
        "rainfall": "Précipitations cumulées",
        "temperature": "Températures moyennes",
        "rainfall_axis": "Précipitations cumulées (mm)",
        "temperature_axis": "Températures moyennes (°C)",
        "year": "Année",
        "saved": "Graphique créé et enregistré",
    },
    "en": {
        "title": "Cumulative rainfall",
        "title_temp": "Mean Temperature",
        "reference": "Baseline",
        "anomaly": "Anomaly",
        "anomaly_axis": "Anomaly (mm)",
        "anomaly_axis_temp": "Anomaly (°C)",
        "rainfall": "Cumulative rainfall",
        "temperature": "Mean Temperature",
        "rainfall_axis": "Cumulative rainfall (mm)",
        "temperature_axis": "Mean Temperature (°C)",
        "year": "Year",
        "saved": "Plot created and saved",
    },
    "mg": {
        "title": "Rotsakorana",
        "title_temp": "Maripana",
        "reference": "Vanim-potoana",
        "anomaly": "Fahasamihafana",
        "anomaly_axis": "Taha mahasamihafa (mm)",
        "anomaly_axis_temp": "Taha mahasamihafa (°C)",
        "rainfall": "Rotsakorana",
        "temperature": "Maripana",
        "rainfall_axis": "Rotsakorana (mm)",
        "temperature_axis": "Maripana ankapobeny (°C)",
        "year": "Taona",
        "saved": "Vita sy voatahiry ny kisarisary",
    },
}

season_mapping = {
    'DJF': [12,1,2],
    'JFM': [1,2,3],
    'FMA': [2,3,4], 
    'MAM': [3,4,5],
    'AMJ': [4,5,6],
    'MJJ': [5,6,7], 
    'JJA': [6,7,8],
    'JAS': [7,8,9],
    'ASO': [8,9,10],
    'SON': [9,10,11], 
    'OND': [10,11,12],
    'ONDJFMA': [10,11,12,1,2,3,4]
}
month_mapping = {
    'janvier': 1, 'fevrier': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6, 
    'juillet': 7, 'aout': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 
    'decembre': 12
}

# Read station data
# Station data shoud have the format csv file, missing values = -99.9
'''
Date,Tmin,Tmax,Rainfall,Tmean
19810101,22.7,31.9,0,27.3
19810102,23,35.3,3.7,29.15
19810103,22,36,0,29
19810104,23.8,34.2,7.8,29
19810105,24.9,31.4,0.6,28.15
19810106,23.3,31,0.3,27.15
19810107,23.6,31.2,10,27.4
19810108,23.8,31.6,4,27.7
19810109,24.4,31.5,67,27.95
...
'''

def read_stn_data(file):
    missing_val = [-999, -99.9, -99, 'NA', 'N/A', '', 'missing']
    url = file
    data = pd.read_csv(file, parse_dates=['Date'], na_values=missing_val)
    df = data.copy()
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    return df

def assign_season_year(row, start_month):
    month = row['Month']
    year = row['Year']

    if month < start_month:
        return year - 1
    else:
        return year

def plot_precip_interannual_variability(
        file, 
        analysis='season', 
        timescale="JFM", 
        language="fr"):
    '''
    Function to plot interannual variability of a month or a season for rainfall
    file: Path or name of a the station data
    analysis: choose between "month" or "season"
    timescale: a season or month , "JFM" or "Jan", must be 3 letter
    language: choose between "fr" for french, "en" for english and "mg" for malagasy

    '''
    analysis = str(analysis).strip().lower()
    timescale = str(timescale).strip().lower()
    language = str(language).strip().lower()

    if language not in TRANSLATIONS: 
        raise ValueError(
            f"Unsupported language {language!r}. "
            "Choose 'fr', 'en', 'mg'. "
        )

    text = TRANSLATIONS[language]
    apply_scientific_plot_style()

    df = read_stn_data(file)
    
    timescale = timescale.upper()

    if analysis == 'season' and timescale in season_mapping:
        if timescale not in season_mapping: 
            raise ValueError(f"La saison {timescale} est invalide. Veuillez entrer une saison valide.")
        
        season_month = season_mapping[timescale]
        start_month = season_month[0]

        df_season = df[df['Month'].isin(season_month)].copy()
        df_season['Season Year'] = df_season.apply(lambda row: assign_season_year(row, start_month), axis=1)
        df_season.dropna(subset=['Season Year'], inplace=True)
        df_season['Season Year'] = df_season['Season Year'].astype(int)
        # baseline climato
        baseline_df = df_season[(df_season['Season Year'] >= 1991) & (df_season['Season Year'] <= 2020)]
        baseline_season_sum = baseline_df.groupby('Season Year')['Rainfall'].sum().mean()
        baseline_monthly_mean = baseline_df.groupby('Month')['Rainfall'].mean()
        df_season_by_year = df_season.groupby('Season Year', as_index=False).agg(
            Sum_season = ('Rainfall', 'sum'))
        df_season_by_year = df_season_by_year[(df_season_by_year['Season Year'] >= 1981)]

        df_season_by_year['Rainfall Anomaly'] = df_season_by_year['Sum_season'] - baseline_season_sum
        df_season['Monthly Mean'] = df_season['Month'].map(baseline_monthly_mean)
        #df_season['Rainfall Anomaly'] = df_season['Rainfall'] - df_season['Monthly Mean']
        df_season['Rainfall Anomaly'] = df_season['Rainfall'] - df_season['Monthly Mean']

        aggregated_data = df_season_by_year.groupby('Season Year').agg(
            Avg_Rainfall=('Sum_season', 'sum'),
            Anomaly=('Rainfall Anomaly', 'sum')
        ).reset_index()
        group_by_column = 'Season Year'
        plot_title = f'Analyse des precipitations saisonnieres ({timescale})'
        baseline_mean_value = aggregated_data['Avg_Rainfall'].mean()
        plot_title = (
            f"{text['title']} ({timescale}) - "
            f"{text['reference']} (1991-2020): "
            f"{baseline_mean_value:.1f} mm"
        )

    elif analysis == 'month':
        timescale = timescale.lower()
        if timescale not in month_mapping: 
            raise ValueError(f"Le mois {timescale} est invalide. Verifiez l'ortographe.")

        month_num = month_mapping[timescale]
        df_month = df[df['Month'] == month_num].copy()
        baseline_df = df_month[(df_month['Year'] >= 1991) & (df_month['Year'] <= 2020)]
        baseline_mean = baseline_df['Rainfall'].mean()

        df_month['Rainfall Anomaly'] = df_month['Rainfall'] - baseline_mean

        aggregated_data = df_month.groupby('Year').agg(
            Avg_Rainfall=('Rainfall', 'sum'),
            Anomaly=('Rainfall Anomaly', 'sum')
        ).reset_index()

        group_by_column = 'Year'
        plot_title = f'Analyse des precipitations mensuelles ({timescale})'
        baseline_mean_value = aggregated_data['Avg_Rainfall'].mean()
        #plot_title = f"Precipitations cumulees ({timescale}) - Baseline Ref. (1991-2020): {baseline_mean_value:.1f} mm"

        plot_title = (
            f"{text['title']} ({timescale}) - "
            f"{text['reference']} (1991-2020): "
            f"{baseline_mean_value:.1f} mm"
        )

    else: 
        raise ValueError("Invalide selection. Veuilez choisir les bonnes valeurs de mois.")

    # plot 
    fig, ax1 = plt.subplots(figsize=(14, 8))
    colors = ['green' if val >=0 else 'orange' for val in aggregated_data['Anomaly']]
#    bars = ax1.bar(aggregated_data[group_by_column], aggregated_data['Anomaly'], 
#                   color=colors, label='Anomalie', alpha=0.8)
    bars = ax1.bar(
        aggregated_data[group_by_column], 
        aggregated_data['Anomaly'], 
        color=colors, 
        label=text['anomaly'],
        alpha=0.8,
    )
    ax2 = ax1.twinx()
#    ax2.plot(aggregated_data[group_by_column].to_numpy(), aggregated_data['Avg_Rainfall']. to_numpy(), 
#             marker='o', color='purple', label='Precipitations cumulees')
    ax2.plot(
        aggregated_data[group_by_column].to_numpy(),
        aggregated_data["Avg_Rainfall"].to_numpy(),
        marker='o',
        color='purple', 
        label=text['rainfall'],
    )

    #ax1.set_ylabel('Anomalie (mm)', fontsize=13)
    ax1.set_ylabel(text['anomaly_axis'], fontsize=13)

    #ax2.set_ylabel('Precipitations cumulees (mm)', fontsize=13, color='purple')
    ax2.set_ylabel(text['rainfall_axis'], fontsize=13)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f} mm"))
    #ax1.set_xlabel('Annee', fontsize=13)
    ax1.set_xlabel(text['year'], fontsize=13)
    set_scientific_title(
        ax1,
        f"{plot_title}",
        fontsize=15,
    )
    style_scientific_grid(ax1)
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
    )
    ax2.tick_params(
        axis="y",
        which="both",
        direction="in",
    )

    for bar in bars:
        height = bar.get_height()
        if height != 0 and not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2, height + np.sign(height)*0.5, 
                     f"{height:+.1f}", ha='center', va='bottom', fontsize=9)

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95), fontsize=11)
    name_fig = f"Interannual_variability_precip_{analysis}_{timescale}_{language}.png"
    plt.tight_layout()
    plt.savefig(name_fig, dpi=300)
    plt.show()
    print(f"{text['saved']}: {name_fig}")

        


def plot_temp_interannual_variability(
        file, 
        analysis='season', 
        timescale="JFM", 
        language="fr"):
    '''
    Function to plot interannual variability of a month or a season for rainfall
    file: Path or name of a the station data
    analysis: choose between "month" or "season"
    timescale: a season or month , "JFM" or "Janvier"
    language: choose between "fr" for french, "en" for english and "mg" for malagasy

    '''
    analysis = str(analysis).strip().lower()
    timescale = str(timescale).strip().lower()
    language = str(language).strip().lower()

    if language not in TRANSLATIONS: 
        raise ValueError(
            f"Unsupported language {language!r}. "
            "Choose 'fr', 'en', 'mg'. "
        )

    text = TRANSLATIONS[language]
    apply_scientific_plot_style()

    df = read_stn_data(file)
    
    timescale = timescale.upper()


    if analysis == 'season' and timescale in season_mapping:
        if timescale not in season_mapping: 
            raise ValueError(f"La saison {timescale} est invalide. Veuillez entrer une saison valide.")
        
        season_month = season_mapping[timescale]
        start_month = season_month[0]

        df_season = df[df['Month'].isin(season_month)].copy()
        df_season['Season Year'] = df_season.apply(lambda row: assign_season_year(row, start_month), axis=1)
        df_season.dropna(subset=['Season Year'], inplace=True)
        df_season['Season Year'] = df_season['Season Year'].astype(int)
        # baseline climato
        baseline_df = df_season[(df_season['Season Year'] >= 1991) & (df_season['Season Year'] <= 2020)]
        baseline_season_sum = baseline_df.groupby('Season Year')['Tmean'].mean().mean()
        baseline_monthly_mean = baseline_df.groupby('Month')['Tmean'].mean()
        df_season_by_year = df_season.groupby('Season Year', as_index=False).agg(
            Mean_season = ('Tmean', 'mean'))
        df_season_by_year = df_season_by_year[(df_season_by_year['Season Year'] >= 1981)]

        df_season_by_year['Temperature Anomaly'] = df_season_by_year['Mean_season'] - baseline_season_sum
        df_season['Monthly Mean'] = df_season['Month'].map(baseline_monthly_mean)
        #df_season['Rainfall Anomaly'] = df_season['Rainfall'] - df_season['Monthly Mean']
        df_season['Temperature Anomaly'] = df_season['Tmean'] - df_season['Monthly Mean']

        aggregated_data = df_season_by_year.groupby('Season Year').agg(
            Mean_temperature=('Mean_season', 'mean'),
            Anomaly=('Temperature Anomaly', 'mean')
        ).reset_index()
        group_by_column = 'Season Year'
        baseline_mean_value = aggregated_data['Mean_temperature'].mean()
        plot_title = (
            f"{text['title_temp']} ({timescale}) - "
            f"{text['reference']} (1991-2020): "
            f"{baseline_mean_value:.1f} °C"
        )

    elif analysis == 'month':
        timescale = timescale.lower()
        if timescale not in month_mapping: 
            raise ValueError(f"Le mois {timescale} est invalide. Verifiez l'ortographe.")

        month_num = month_mapping[timescale]
        df_month = df[df['Month'] == month_num].copy()
        baseline_df = df_month[(df_month['Year'] >= 1991) & (df_month['Year'] <= 2020)]
        baseline_mean = baseline_df['Tmean'].mean()

        df_month['Temperature Anomaly'] = df_month['Tmean'] - baseline_mean

        aggregated_data = df_month.groupby('Year').agg(
            Mean_Temperature=('Tmean', 'mean'),
            Anomaly=('Temperature Anomaly', 'mean')
        ).reset_index()

        group_by_column = 'Year'
        baseline_mean_value = aggregated_data['Mean_Temperature'].mean()
        #plot_title = f"Precipitations cumulees ({timescale}) - Baseline Ref. (1991-2020): {baseline_mean_value:.1f} mm"

        plot_title = (
            f"{text['title_temp']} ({timescale}) - "
            f"{text['reference']} (1991-2020): "
            f"{baseline_mean_value:.1f} °C"
        )

    else: 
        raise ValueError("Invalide selection. Veuilez choisir les bonnes valeurs de mois.")

    # plot 
    fig, ax1 = plt.subplots(figsize=(14, 8))
    colors = ['red' if val >=0 else 'blue' for val in aggregated_data['Anomaly']]
#    bars = ax1.bar(aggregated_data[group_by_column], aggregated_data['Anomaly'], 
#                   color=colors, label='Anomalie', alpha=0.8)
    bars = ax1.bar(
        aggregated_data[group_by_column], 
        aggregated_data['Anomaly'], 
        color=colors, 
        label=text['anomaly'],
        alpha=0.8,
    )
    ax2 = ax1.twinx()
#    ax2.plot(aggregated_data[group_by_column].to_numpy(), aggregated_data['Avg_Rainfall']. to_numpy(), 
#             marker='o', color='purple', label='Precipitations cumulees')
    ax2.plot(
        aggregated_data[group_by_column].to_numpy(),
        aggregated_data["Mean_Temperature"].to_numpy(),
        marker='o',
        color='purple', 
        label=text['temperature'],
    )

    #ax1.set_ylabel('Anomalie (mm)', fontsize=13)
    ax1.set_ylabel(text['anomaly_axis_temp'], fontsize=13)

    #ax2.set_ylabel('Precipitations cumulees (mm)', fontsize=13, color='purple')
    ax2.set_ylabel(text['temperature_axis'], fontsize=13)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f} °C"))
    #ax1.set_xlabel('Annee', fontsize=13)
    ax1.set_xlabel(text['year'], fontsize=13)
    set_scientific_title(
        ax1,
        f"{plot_title}",
        fontsize=15,
    )
    style_scientific_grid(ax1)
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
    )
    ax2.tick_params(
        axis="y",
        which="both",
        direction="in",
    )

    for bar in bars:
        height = bar.get_height()
        if height != 0 and not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2, height + np.sign(height)*0.08, 
                     f"{height:+.1f}", ha='center', va='bottom', fontsize=9)

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95), fontsize=11)
    name_fig = f"Interannual_variability_temperature_{analysis}_{timescale}_{language}.png"
    plt.tight_layout()
    plt.savefig(name_fig, dpi=300)
    plt.show()
    print(f"{text['saved']}: {name_fig}")
