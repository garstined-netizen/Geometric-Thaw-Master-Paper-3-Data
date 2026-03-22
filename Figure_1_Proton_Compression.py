import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

# =============================================================================
# 1. Setup and Fundamental Constants (From Geometric Thaw / SVCF Framework)
# =============================================================================
# Radii (fm)
r_p_e = 0.8775     # Un-damped electronic proton radius (CODATA)
r_p_mu = 0.84087   # Damped muonic proton radius (CREMA)

# Masses (MeV/c^2)
m_e = 0.511
m_mu = 105.66

# =============================================================================
# 2. Mathematical Wave Modeling (Fluid-Damped Harmonic Oscillator)
# =============================================================================
def topological_wave(X, Y, radius_boundary, damping_factor=0.0):
    """
    Simulates the proton as a resonant acoustic standing wave (topological defect).
    Amplitude is constricted by the damping factor (metric viscosity).
    """
    R = np.sqrt(X**2 + Y**2)
    # The acoustic resonance is modeled as a decaying spherical Bessel-like function
    # Damping compresses the spatial extent of the wave envelope
    frequency = 3.0 + damping_factor 
    envelope = np.exp(-(R**2) / (radius_boundary**2))
    wave = np.cos(frequency * R) * envelope
    return wave

# Create spatial meshgrid (femtometer scale)
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
R_grid = np.sqrt(X**2 + Y**2)

# Generate Z data for both states
Z_electronic = topological_wave(X, Y, r_p_e, damping_factor=0.0)
Z_muonic = topological_wave(X, Y, r_p_mu, damping_factor=1.5)

# Generate localized metric shear (viscosity) wake for the muon
# Peaks sharply at the origin (inverse cube scaling smoothed for visualization)
viscosity_wake = np.exp(- (R_grid**2) / 0.3) 

# =============================================================================
# 3. Figure Architecture & Plotting
# =============================================================================
fig = plt.figure(figsize=(14, 10), dpi=150)
fig.suptitle('Figure 1: Viscoelastic Metric Drag & Proton Radial Compression', 
             fontsize=18, weight='bold', y=0.95)

# Create a Grid: Top row has two 3D plots, Bottom row spans the width for scatter
gs = GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.1)

# --- PANEL A: Un-damped Electronic Proton (3D) ---
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_electronic, cmap='coolwarm', alpha=0.9, 
                         linewidth=0, antialiased=True)
ax1.set_title(r'A. Electronic Proton (Superfluid Vacuum, $\eta=0$)', fontsize=12, weight='bold')
ax1.set_zlim(-1.2, 1.2)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.view_init(elev=25, azim=45)
# Add a flat zero-viscosity base to represent the pristine vacuum
ax1.contourf(X, Y, np.zeros_like(Z_electronic), zdir='z', offset=-1.2, cmap='Blues', alpha=0.3)

# --- PANEL B: Damped Muonic Proton (3D) ---
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_muonic, cmap='coolwarm', alpha=0.9, 
                         linewidth=0, antialiased=True)
ax2.set_title(r'B. Muonic Proton (Melted Metric Wake, $\eta > 0$)', fontsize=12, weight='bold')
ax2.set_zlim(-1.2, 1.2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.view_init(elev=25, azim=45)
# Project the highly viscous "thawed" wake heatmap directly beneath the compressed wave
wake = ax2.contourf(X, Y, viscosity_wake, zdir='z', offset=-1.2, cmap='magma', alpha=0.8)
cbar = fig.colorbar(wake, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label(r'Local Metric Shear Viscosity ($\eta_{shear}$)', rotation=270, labelpad=15)

# --- PANEL C: Lepton Mass vs. Compression Correlation (Scatter Plot) ---
ax3 = fig.add_subplot(gs[1, :])

# Lepton masses (log scale for visualization) and corresponding radii
masses = [m_e, m_mu]
radii = [r_p_e, r_p_mu]
labels = ['Electronic Hydrogen\n(Un-damped, 0.8775 fm)', 
          'Muonic Hydrogen\n(FDHO Compressed, 0.8408 fm)']

# Plot the deterministic linear decay model connecting the two states
x_line = np.linspace(0.1, 150, 100)
# Simple logarithmic fit to bridge the two points visually for the correlation
slope = (r_p_mu - r_p_e) / (np.log(m_mu) - np.log(m_e))
intercept = r_p_e - slope * np.log(m_e)
y_line = slope * np.log(x_line) + intercept

ax3.plot(x_line, y_line, '--', color='gray', lw=2, zorder=1, 
         label='FDHO Subatomic Viscosity Trend (Geometric Thaw)')

# Overlay the exact data points
colors = ['blue', 'red']
for i in range(2):
    ax3.scatter(masses[i], radii[i], s=150, c=colors[i], zorder=3, edgecolors='black')
    ax3.annotate(labels[i], (masses[i], radii[i]), 
                 textcoords="offset points", xytext=(10, 10), ha='left', fontsize=11)

ax3.set_xscale('log')
ax3.set_xlabel(r'Orbital Lepton Mass ($MeV/c^2$) - Log Scale', fontsize=12, weight='bold')
ax3.set_ylabel(r'Extracted Proton Radius (fm)', fontsize=12, weight='bold')
ax3.set_title(r'C. Deterministic Correlation: Mass-Induced Viscosity vs. Radial Compression', 
              fontsize=13, weight='bold')

ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='lower left', fontsize=11)

# Annotate the 0.0366 fm discrepancy visually
ax3.annotate(r'$\Delta r = 0.0366$ fm (Viscoelastic Drag)', 
             xy=(10, 0.858), xytext=(2, 0.865),
             arrowprops=dict(facecolor='black', arrowstyle='<->', lw=1.5),
             fontsize=12, color='darkred', weight='bold')

# Save and Show
plt.tight_layout()
plt.subplots_adjust(top=0.90) # Adjust title spacing
plt.savefig("Figure_1_Proton_Compression.pdf", format='pdf', bbox_inches='tight')
plt.show()