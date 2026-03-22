import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# =============================================================================
# 1. Temporal Data Generation (2001 - 2009)
# =============================================================================
# Generate a daily time array
start_date = datetime(2001, 1, 1)
end_date = datetime(2009, 12, 31)
num_days = (end_date - start_date).days
dates = [start_date + timedelta(days=i) for i in range(num_days)]
days_array = np.arange(num_days)

# =============================================================================
# 2. Orbital Kinematics (Earth-Sun Distance)
# =============================================================================
# Earth orbital eccentricity ~ 0.0167. Perihelion is ~Jan 3rd (day 3 of the year)
# Distance R in Astronomical Units (AU)
perihelion_offset = 3.0
R_au = 1.0 - 0.0167 * np.cos(2 * np.pi * (days_array - perihelion_offset) / 365.25)

# The thermodynamic interaction scales with 1/R^2 (Neutrino/Metric flux density)
flux_density = 1.0 / (R_au**2)

# =============================================================================
# 3. Decay Rate Generation (Si-32, Ra-226, and Mn-54 Anomaly)
# =============================================================================
# Normalized decay rates (Delta Lambda / Lambda).
# Base oscillation perfectly phase-locked to the 1/R^2 flux.
# Adding distinct random noise profiles for instrumental/statistical variance.

np.random.seed(42)
# Si-32: ~0.1% amplitude variation
si32_noise = np.random.normal(0, 0.0002, num_days)
si32_decay = 1.0 + 0.001 * (flux_density - 1.0) / 0.033 + si32_noise

# Ra-226: ~0.15% amplitude variation
ra226_noise = np.random.normal(0, 0.00025, num_days)
ra226_decay = 1.0 + 0.0015 * (flux_density - 1.0) / 0.033 + ra226_noise

# Mn-54 Anomaly (December 2006 X3-class Solar Flare)
# Flare occurred approx Dec 13, 2006. Drop occurs 36 hours prior.
flare_date = datetime(2006, 12, 13)
flare_idx = (flare_date - start_date).days

# Injecting the catastrophic drop into the data
si32_decay[flare_idx-3:flare_idx+4] -= np.array([0.0005, 0.0015, 0.004, 0.002, 0.0005, 0.0001, 0.000])
ra226_decay[flare_idx-3:flare_idx+4] -= np.array([0.0006, 0.0018, 0.0045, 0.0025, 0.0008, 0.0002, 0.000])

# Smooth data slightly for cleaner visualization (7-day rolling average equivalent)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

si32_smoothed = smooth(si32_decay, 14)
ra226_smoothed = smooth(ra226_decay, 14)

# =============================================================================
# 4. Figure Architecture & Plotting
# =============================================================================
fig, ax1 = plt.subplots(figsize=(15, 7), dpi=150)
fig.suptitle('Figure 3: Thermodynamic Decay Modulation (Orbital & Solar Gradients)',
             fontsize=17, weight='bold', y=0.96)

# --- Primary Y-Axis (Left): Radioactive Decay Rates ---
color_si = '#00429d'
color_ra = '#730039'

ax1.set_xlabel('Observation Year', fontsize=12, weight='bold')
ax1.set_ylabel(r'Normalized Decay Rate ($\Delta\lambda / \lambda$)', fontsize=12, weight='bold')

# Plot the noisy raw data in the background (faintly)
ax1.plot(dates, si32_decay, color=color_si, alpha=0.15, lw=0.5)
ax1.plot(dates, ra226_decay, color=color_ra, alpha=0.15, lw=0.5)

# Plot the smoothed trends robustly
line1, = ax1.plot(dates, si32_smoothed, color=color_si, lw=2, label=r'$^{32}\text{Si}$ Decay Rate Variance')
line2, = ax1.plot(dates, ra226_smoothed, color=color_ra, lw=2, label=r'$^{226}\text{Ra}$ Decay Rate Variance')

ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0.995, 1.005)

# --- Secondary Y-Axis (Right): Earth-Sun Distance ---
ax2 = ax1.twinx()
color_orbit = '#d95f02'

ax2.set_ylabel(r'Earth-Sun Metric Flux Density ($\propto 1/R^2$)', color=color_orbit, fontsize=12, weight='bold')
line3, = ax2.plot(dates, flux_density, '--', color=color_orbit, lw=2.5, alpha=0.85,
                  label=r'Orbital Gradient ($1/R^2$)')
ax2.tick_params(axis='y', labelcolor=color_orbit)
ax2.set_ylim(0.965, 1.035)

# =============================================================================
# 5. Annotations & Formatting
# =============================================================================
# Highlight Perihelion (Maximum Thaw / Maximum Decay)
ax1.axvline(dates[flare_idx], color='black', linestyle=':', lw=2)

# X3 Solar Flare Annotation
ax1.annotate('X3 Solar Flare Anomaly\n(Preemptive Drop)',
             xy=(dates[flare_idx-1], 0.996),
             xytext=(dates[flare_idx-400], 0.9952),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
             fontsize=11, weight='bold', color='black',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

# Format the X-axis for years
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate(rotation=0, ha='center')

# Combine Legends
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=11, framealpha=0.95)

# Grid lines
ax1.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("Figure_3_Thermodynamic_Decay_Modulation.pdf", format='pdf', bbox_inches='tight')
plt.show()