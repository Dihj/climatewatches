import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -------------------------------------------------------------------------
# 1. GENERATE ICON IMAGE ARRAYS (Cyclone, Flood, Drought)
# -------------------------------------------------------------------------
def create_cyclone_icon(size=40):
    """Draws a spiral cyclone icon (RGBA array)."""
    img = np.zeros((size, size, 4))
    cx, cy = size // 2, size // 2
    for r in range(2, size // 2 - 2):
        angle = r * 0.8
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            img[y-1:y+2, x-1:x+2] = [0.1, 0.1, 0.2, 0.9]  # Dark navy swirl
    return img

def create_flood_icon(size=40):
    """Draws water waves icon (RGBA array)."""
    img = np.zeros((size, size, 4))
    x = np.linspace(0, 4 * np.pi, size)
    for row_y, offset in [(15, 0), (25, np.pi/2)]:
        y = (row_y + 4 * np.sin(x + offset)).astype(int)
        for i, y_val in enumerate(y):
            if 0 <= y_val < size:
                img[y_val-1:y_val+2, i] = [0.1, 0.4, 0.8, 0.95]  # Blue waves
    return img

def create_drought_icon(size=40):
    """Draws a sun/drought icon (RGBA array)."""
    img = np.zeros((size, size, 4))
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # Sun disk
    img[dist <= 7] = [0.85, 0.45, 0.1, 0.95]
    # Sun rays
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for r in range(9, 16):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = [0.85, 0.45, 0.1, 0.95]
    return img

ICON_MAP = {
    'Tropical cyclone': create_cyclone_icon(),
    'Storm': create_cyclone_icon(),
    'Flood': create_flood_icon(),
    'Drought': create_drought_icon()
}

# -------------------------------------------------------------------------
# 2. LOAD & PROCESS DATASETS
# -------------------------------------------------------------------------
df_disasters = pd.read_csv('../data/public_emdat_data_Madagascar.csv')
df_crops = pd.read_csv('../data/faostat_data.csv')

# Crop Data
rice_df = df_crops[
    (df_crops['Item'].str.contains('Rice', case=False, na=False)) & 
    (df_crops['Element'].str.contains('Production', case=False, na=False))
].copy()
rice_annual = rice_df.groupby('Year')['Value'].sum().reset_index()

# Disaster Data
hydro_disasters = df_disasters[
    df_disasters['Disaster Group'].isin(['Natural']) & 
    df_disasters['Disaster Subgroup'].isin(['Meteorological', 'Hydrological', 'Climatological'])
].copy()

disaster_counts = hydro_disasters.groupby('Start Year')['DisNo.'].count().reset_index()
disaster_counts.rename(columns={'Start Year': 'Year', 'DisNo.': 'Disaster_Count'}, inplace=True)

# Select highest-impact disaster for key annual callouts
hydro_disasters['Impact_Score'] = hydro_disasters['Total Affected'].fillna(0) + hydro_disasters['Total Damage (\'000 US$)'].fillna(0)
top_annual_events = hydro_disasters.sort_values(by='Impact_Score', ascending=False).groupby('Start Year').first().reset_index()

# Merge years 2000-2024
merged_df = pd.merge(rice_annual, disaster_counts, on='Year', how='left').fillna(0)
merged_df = merged_df[(merged_df['Year'] >= 2000) & (merged_df['Year'] <= 2024)]

# -------------------------------------------------------------------------
# 3. DUAL-AXIS PLOTTING
# -------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(16, 8))

# Left Axis: Production Line
color_line = '#15803d'  # Green
ax1.plot(merged_df['Year'], merged_df['Value'], color=color_line, marker='o', linewidth=2.5, label='Rice production')
ax1.set_ylabel('Rice production (metric tonnes)', color=color_line, fontweight='bold', fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_line)

# Add upper space margin for icons
y_max_prod = merged_df['Value'].max() * 1.45 if merged_df['Value'].max() > 0 else 100
ax1.set_ylim(0, y_max_prod)

# Value annotations for line data points
for x, y in zip(merged_df['Year'], merged_df['Value']):
    if y > 0:
        ax1.annotate(f'{int(y):,}', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=8, color=color_line, fontweight='bold')

# Right Axis: Disaster Count Bars
ax2 = ax1.twinx()
color_bars = '#1e3a8a'  # Dark Navy
bars = ax2.bar(merged_df['Year'], merged_df['Disaster_Count'], color=color_bars, width=0.45, alpha=0.85, label='Disaster count')
ax2.set_ylabel('Number of hydrometeorological disasters', color=color_bars, fontweight='bold', fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_bars)

# Numeric labels on top of bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, color=color_bars, fontweight='bold')

# -------------------------------------------------------------------------
# 4. DRAW ICON MARKS & EVENT LABELS
# -------------------------------------------------------------------------
for _, row in top_annual_events.iterrows():
    yr = int(row['Start Year'])
    if 2000 <= yr <= 2024:
        prod_val = merged_df.loc[merged_df['Year'] == yr, 'Value'].values
        if len(prod_val) > 0 and prod_val[0] > 0:
            y_pos = prod_val[0]
            
            subtype = row['Disaster Subtype']
            event_name = row['Event Name'] if pd.notna(row['Event Name']) else subtype
            icon_img = ICON_MAP.get(subtype, create_cyclone_icon())
            
            # Target annotation position above production line
            y_target = y_pos + (y_max_prod * 0.18)
            
            # Stem line from line point to icon base
            ax1.plot([yr, yr], [y_pos, y_target - (y_max_prod * 0.04)], color='#94a3b8', linestyle=':', lw=1.2)
            
            # Insert Icon
            imagebox = OffsetImage(icon_img, zoom=0.7)
            ab = AnnotationBbox(imagebox, (yr, y_target), frameon=False)
            ax1.add_artist(ab)
            
            # Event label below icon
            ax1.text(yr, y_target - (y_max_prod * 0.05), f"{yr}\n{event_name}", 
                     ha='center', va='top', fontsize=7.5, fontweight='bold', color='#7f1d1d')

# -------------------------------------------------------------------------
# 5. FORMATTING & DISPLAY
# -------------------------------------------------------------------------
ax1.set_xticks(merged_df['Year'])
ax1.set_xticklabels(merged_df['Year'].astype(int), rotation=45)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.title('Madagascar: Rice Production vs. Major Hydrometeorological Disasters', fontsize=13, fontweight='bold', pad=25, color='#1e293b')

plt.tight_layout()
plt.show()
