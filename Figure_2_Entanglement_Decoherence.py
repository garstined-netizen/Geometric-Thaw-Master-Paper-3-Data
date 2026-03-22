import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# =============================================================================
# 1. Geographic Data & Micius Orbital Track
# =============================================================================
# Ground Station Coordinates (Longitude, Latitude)
stations = {
    'Nanshan (Low Shear)': (87.176, 43.475),
    'Delingha (Med Shear)': (97.727, 37.379),
    'Lijiang (High Shear)': (100.029, 26.693)
}

# Simulated Sun-Synchronous Orbital Track (approx 97.4 deg inclination)
orbit_lat = np.linspace(60, 10, 100)
orbit_lon = 95.0 - (60 - orbit_lat) * 0.15 # Slight drift over China

# =============================================================================
# 2. Lense-Thirring Shear Gradient Model
# =============================================================================
# Generate a global grid
lon_grid = np.linspace(-180, 180, 360)
lat_grid = np.linspace(-90, 90, 180)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

# Lense-Thirring Shear is maximized at the equator (latitude 0)
# We model this visually as a steep drop-off towards the poles
shear_viscosity = np.cos(np.radians(Lat))**6 

# =============================================================================
# 3. Phase-Space Data (From the Viscous Interference Table)
# =============================================================================
# Gamma values (Viscoelastic Path Integral x 10^-6)
gamma_data = np.array([1.12, 2.84, 4.37, 5.82, 6.91, 9.55])
# Observed S-parameters
s_observed = np.array([2.57, 2.45, 2.32, 2.18, 2.05, 1.94])
# Classical Expected S-parameters (Hovering near 2.3 - 2.6)
s_classical = np.array([2.58, 2.50, 2.44, 2.41, 2.36, 2.38])
# Corresponding Targets for the data points
targets = ['Nanshan', 'Delingha', 'Delingha', 'Lijiang', 'Lijiang', 'Lijiang']

# =============================================================================
# 4. Figure Architecture & Plotting
# =============================================================================
fig = plt.figure(figsize=(14, 12), dpi=150)
fig.suptitle('Figure 2: Viscoelastic Decoherence in Satellite-to-Ground Entanglement', 
             fontsize=18, weight='bold', y=0.96)

gs = GridSpec(2, 1, height_ratios=[1.8, 1], hspace=0.2)

# --- PANEL A: Geographic Lense-Thirring Shear Map ---
# Using an Orthographic projection focused on China
ax_map = fig.add_subplot(gs[0], projection=ccrs.Orthographic(central_longitude=95.0, central_latitude=35.0))
ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
ax_map.add_feature(cfeature.OCEAN, facecolor='white')
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax_map.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)

# Overlay the "Thawed" Spacetime Viscosity Gradient
shear_plot = ax_map.contourf(Lon, Lat, shear_viscosity, 100, transform=ccrs.PlateCarree(), 
                             cmap='jet', alpha=0.4)
cbar = plt.colorbar(shear_plot, ax=ax_map, shrink=0.7, pad=0.05)
cbar.set_label(r'Normalized Local Metric Shear ($\eta_{EH}$)', rotation=270, labelpad=15)

# Plot Ground Stations
colors = {'Nanshan': 'blue', 'Delingha': 'purple', 'Lijiang': 'red'}
for name, coords in stations.items():
    ax_map.plot(coords[0], coords[1], marker='^', color='black', markersize=10, 
                transform=ccrs.PlateCarree(), zorder=5)
    ax_map.text(coords[0] + 2, coords[1], name, transform=ccrs.PlateCarree(), 
                fontsize=11, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Plot Micius Orbital Track
ax_map.plot(orbit_lon, orbit_lat, color='green', linewidth=2, linestyle='--', 
            transform=ccrs.PlateCarree(), label='Micius (QUESS) LEO Ground Track')

# Draw simulated quantum tethers (lines from satellite path to stations)
# Approximating beam paths at different latitudes
ax_map.plot([orbit_lon[10], stations['Nanshan (Low Shear)'][0]], [orbit_lat[10], stations['Nanshan (Low Shear)'][1]], 
            color='blue', linestyle=':', lw=2, transform=ccrs.PlateCarree())
ax_map.plot([orbit_lon[50], stations['Delingha (Med Shear)'][0]], [orbit_lat[50], stations['Delingha (Med Shear)'][1]], 
            color='purple', linestyle=':', lw=2, transform=ccrs.PlateCarree())
ax_map.plot([orbit_lon[80], stations['Lijiang (High Shear)'][0]], [orbit_lat[80], stations['Lijiang (High Shear)'][1]], 
            color='red', linestyle=':', lw=2, transform=ccrs.PlateCarree())

ax_map.set_title('A. Topographic Map of Earth\'s Lense-Thirring Boundary Layer & Micius Downlinks', fontsize=14)
ax_map.legend(loc='upper left')

# --- PANEL B: Vortex Decoherence Plot (Phase-Space Regression) ---
ax_data = fig.add_subplot(gs[1])

# Plot the Classical Expected S-parameter (Null Hypothesis)
ax_data.plot(gamma_data, s_classical, 's--', color='gray', label='Classical Expected S (Null Hypothesis)', alpha=0.7)

# Plot the Observed Anomalous S-parameter
ax_data.plot(gamma_data, s_observed, 'ko-', lw=2, label='Observed S (Micius Telemetry)')

# Color-code the scatter points based on the target station
for i in range(len(gamma_data)):
    color = colors[targets[i]]
    ax_data.scatter(gamma_data[i], s_observed[i], color=color, s=100, zorder=5, edgecolors='black')

# Draw the Classical Bell Limit Threshold (S = 2.0)
ax_data.axhline(2.0, color='red', linestyle='--', lw=2, label='Classical Bell Limit (S = 2.0)')
ax_data.fill_between(gamma_data, 1.8, 2.0, color='red', alpha=0.1)

# Annotations & Formatting
ax_data.set_xlabel(r'Viscous Interference Score $\Gamma_{decoherence}$ ($10^{-6}$ Pa$\cdot$s)', fontsize=12)
ax_data.set_ylabel('CHSH Bell Parameter ($S$)', fontsize=12)
ax_data.set_title('B. Phase-Space Regression: Entanglement Rupture vs. Metric Drag', fontsize=14)
ax_data.set_ylim(1.85, 2.7)
ax_data.grid(True, linestyle=':', alpha=0.6)
ax_data.legend(loc='upper right', fontsize=10)

# Highlight the rupture point
ax_data.annotate('Quantized Vortex Rupture\n(Lijiang Downlink)', xy=(9.55, 1.94), xytext=(7.0, 1.90),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11, weight='bold', color='darkred')

plt.tight_layout()
plt.savefig("Figure_2_Entanglement_Decoherence.pdf", format='pdf', bbox_inches='tight')
plt.show()