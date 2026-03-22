import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# 1. Kinematic Data Generation (Galactic Rotation)
# =============================================================================
# Radii from galactic center (kpc)
r = np.linspace(0.1, 30, 500)

# Velocity constants (km/s)
V_flat = 240.0         # The observed flat rotation plateau
v_c = 130.0            # The Landau Critical Velocity (Cosmic Superfluid threshold)
scale_length = 2.5     # Galactic disk scale length

# 1a. Observed Velocity (Flat Rotation Curve)
# Rapid rise in the core, flattening out in the halo
v_obs = V_flat * (1 - np.exp(-r / (scale_length * 0.8)))

# 1b. Baryonic Expectation (Keplerian Decline)
# Purely luminous matter (bulge + disk), falls off as 1/sqrt(r) in the halo
v_baryon = 180.0 * (r / scale_length) * np.exp(-r / scale_length) + (80.0 / np.sqrt(r + 0.5))

# Find the critical radius (R_c) where v_obs crosses v_c
r_c_index = np.argmax(v_obs > v_c)
r_c = r[r_c_index]

# =============================================================================
# 2. Fluid-Dynamic Spatial Mapping (2D Cross-Section)
# =============================================================================
# Generate a 2D spatial grid (x, y in kpc)
grid_size = 30
x = np.linspace(-grid_size, grid_size, 500)
y = np.linspace(-grid_size, grid_size, 500)
X, Y = np.meshgrid(x, y)
R_grid = np.sqrt(X**2 + Y**2)

# Calculate velocity field across the 2D grid
V_field = V_flat * (1 - np.exp(-R_grid / (scale_length * 0.8)))

# Calculate Viscosity (eta) based on Spacetime Creep
# eta = 0 if V < v_c (Superfluid), eta > 0 if V > v_c (Normal Fluid)
eta_field = np.zeros_like(V_field)
mask_thawed = V_field > v_c
eta_field[mask_thawed] = V_field[mask_thawed] - v_c

# Normalize for the colormap
eta_norm = eta_field / np.max(eta_field)

# Create a custom colormap: Deep blue for Superfluid, striking Red/Orange for Thawed
colors = [(0.0, 0.1, 0.4), (0.1, 0.3, 0.7), (0.2, 0.6, 0.9),  # Superfluid Blues
          (1.0, 0.8, 0.0), (1.0, 0.4, 0.0), (0.7, 0.0, 0.0)]  # Thawed Oranges/Reds
cmap_thaw = LinearSegmentedColormap.from_list('GeometricThaw', colors, N=256)

# =============================================================================
# 3. Figure Architecture & Plotting
# =============================================================================
fig = plt.figure(figsize=(16, 7), dpi=150)
fig.suptitle('Figure 2: Dark Matter as Spacetime Creep (Vacuum Phase Transition)',
             fontsize=18, weight='bold', y=0.98)

gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.15)

# --- PANEL A: Galactic Rotation Curve ---
ax1 = fig.add_subplot(gs[0])

# Shade the vacuum phases
ax1.axvspan(0, r_c, color='#1b4f93', alpha=0.15, label=r'Superfluid Vacuum ($\eta = 0$)')
ax1.axvspan(r_c, 30, color='#d32d2d', alpha=0.15, label=r'Thawed Metric ($\eta > 0$)')

# Plot kinematic curves
ax1.plot(r, v_obs, '-', color='black', lw=3, label=r'Observed Velocity ($v_{obs}$)')
ax1.plot(r, v_baryon, '--', color='gray', lw=2.5, label=r'Baryonic Expectation (Keplerian)')

# Plot Landau Critical Velocity Threshold
ax1.axhline(v_c, color='darkred', linestyle=':', lw=2.5, label=r'Landau Critical Velocity ($v_c$)')

# Annotate the Critical Radius
ax1.plot(r_c, v_c, 'ko', markersize=8, zorder=5)
ax1.annotate(rf'$R_c \approx {r_c:.1f}$ kpc', xy=(r_c, v_c), xytext=(r_c + 1, v_c - 20),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=11, weight='bold')

ax1.set_xlabel('Radius from Galactic Center (kpc)', fontsize=12, weight='bold')
ax1.set_ylabel('Orbital Velocity (km/s)', fontsize=12, weight='bold')
ax1.set_title('A. Anomalous Kinematics & The Critical Threshold', fontsize=14, weight='bold')
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 300)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)

# --- PANEL B: 2D Fluid-Dynamic Heatmap ---
ax2 = fig.add_subplot(gs[1])

# Plot the spacetime viscosity heatmap
heatmap = ax2.contourf(X, Y, eta_norm, levels=100, cmap=cmap_thaw, zorder=1)

# Overlay a synthetic "Galactic Disk" (spiral arms) for astrophysical context
theta = np.linspace(0, 6 * np.pi, 2000)
r_spiral = np.linspace(0.2, 25, 2000)
# Arm 1
x_s1 = r_spiral * np.cos(theta + np.log(r_spiral))
y_s1 = r_spiral * np.sin(theta + np.log(r_spiral))
# Arm 2
x_s2 = r_spiral * np.cos(theta + np.log(r_spiral) + np.pi)
y_s2 = r_spiral * np.sin(theta + np.log(r_spiral) + np.pi)

# Scatter the "stars" over the fluid
ax2.scatter(x_s1, y_s1, s=1.5, color='white', alpha=0.6, zorder=3)
ax2.scatter(x_s2, y_s2, s=1.5, color='white', alpha=0.6, zorder=3)
# Core glow
core_glow = plt.Circle((0, 0), r_c, color='white', alpha=0.2, zorder=2)
ax2.add_patch(core_glow)

# Draw the Critical Radius boundary
circle_rc = plt.Circle((0, 0), r_c, color='cyan', fill=False, linestyle='--', lw=2, zorder=4)
ax2.add_patch(circle_rc)
ax2.text(r_c + 0.5, r_c + 0.5, r'Phase Boundary ($R_c$)', color='cyan', fontsize=11, weight='bold', zorder=5)

ax2.set_aspect('equal')
ax2.set_xlim(-grid_size, grid_size)
ax2.set_ylim(-grid_size, grid_size)
ax2.set_xlabel('Distance X (kpc)', fontsize=12, weight='bold')
ax2.set_ylabel('Distance Y (kpc)', fontsize=12, weight='bold')
ax2.set_title('B. Spacetime Creep: Topographical Phase Heatmap', fontsize=14, weight='bold')

# Colorbar for the Viscosity
cbar = plt.colorbar(heatmap, ax=ax2, shrink=0.8, pad=0.03)
cbar.set_label(r'Normalized Vacuum Viscosity ($\eta_{vac}$)', rotation=270, labelpad=20, weight='bold', fontsize=11)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['0 (Superfluid)', 'Low Drag', 'Medium Drag', 'High Drag', 'Max Viscosity'])

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("Figure_2_Dark_Matter_Spacetime_Creep.pdf", format='pdf', bbox_inches='tight')
plt.show()