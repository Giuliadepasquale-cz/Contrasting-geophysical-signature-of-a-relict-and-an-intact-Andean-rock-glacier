import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import pygimli as pg
from fpinv import add_inner_title, logFormat, rst_cov, set_style
from pygimli.mplviewer import drawModel


######################################################################################## FUNCTIONS ##########################################################################################################
def update_ticks(cb, label="", logScale=False, cMin=None, cMax=None):
    cb.set_ticks([cMin, cMax])
    ticklabels = cb.ax.yaxis.get_ticklabels()
    for i, tick in enumerate(ticklabels):
        if i == 0:
            tick.set_verticalalignment("bottom")
        if i == len(ticklabels) - 1:
            tick.set_verticalalignment("top")
    cb.ax.annotate(label, xy=(1, 0.5), xycoords='axes fraction', xytext=(10, 0), textcoords='offset pixels', horizontalalignment='center', verticalalignment='center', rotation=90, fontsize=fs, fontweight="regular")
    if logScale:
        for lab in cb.ax.yaxis.get_minorticklabels():
            lab.set_visible(False)


def lim(data):
    """Return appropriate colorbar limits."""
    data = np.array(data)
    print("dMin", data.min(), "dMax", data.max())
    if data.min() < 0:
        dmin = 0.0
    else:
        dmin = np.around(data.min(), 2)
    dmax = np.around(data.max(), 2)
    kwargs = {"cMin": dmin, "cMax": dmax}
    return kwargs


def draw(ax, mesh, model, **kwargs):
    model = np.array(model)
    if not np.isclose(model.min(), 0.0, atol=9e-3) and (model < 0).any():
        model = np.ma.masked_where(model < 0, model)
        model = np.ma.masked_where(model > 1, model)

    if "coverage" in kwargs:
        model = np.ma.masked_where(kwargs["coverage"] == 0, model)
    gci = drawModel(ax, mesh, model, rasterized=True, nLevs=2, **kwargs)
    return gci


def minmax(data):
    """Return minimum and maximum of data as a 2-line string."""
    tmp = np.array(data)
    print("max", tmp.max())
    if np.isclose(tmp.min(), 0, atol=9e-3):
        min = 0
    else:
        min = tmp.min()
    if np.max(tmp) > 10 and np.max(tmp) < 1e4:
        return "min: %d | max: %d" % (min, tmp.max())
    if np.max(tmp) > 1e4:
        return "min: %d" % min + " | max: " + logFormat(tmp.max())
    else:
        return "min: %.2f | max: %.2f" % (min, tmp.max())

def to_sat(fw, fi, fa, fr):
    phi = 1 - fr
    return fw / phi, fi / phi, fa / phi
#############################################################################################################################################################################################################

plt.rcParams.update({'font.size':20})

# Load data
mesh = pg.load("paraDomain.bms")
joint = np.load("joint_inversion_n2_m13.npz")
sensors = np.loadtxt("sensors.npy")
veljoint, rhojoint, faj, fij, fwj, frj, maskj = joint["vel"], joint["rho"], joint["fa"], joint["fi"], joint["fw"], joint["fr"], joint["mask"]
#fwj, fij, faj = to_sat(fwj, fij, faj, frj)
cov = rst_cov(mesh, np.loadtxt("rst_coverage.dat"))

labels = ["v (m/s)", r" $\rho$ ($\Omega$m)"]
labels.extend([r"f$_{\rm %s}$" % x for x in "wiar"])
long_labels = ["Velocity", "Resistivity", "Water content", "Ice content", "Air content", "Rock content"]
cmaps = ["viridis", "Spectral_r", "Blues", "Purples", "Greens", "Oranges"]
datas = [veljoint, rhojoint, fwj, fij, faj, frj]


################################################################################################## FIGURES ##################################################################################################
#0) VELOCITY MODEL 
logScale=False
lims={"cMin": 400, "cMax": 4000}
ax0=plt.subplot(1,1,1)
im0=draw(ax0, mesh, datas[0], **lims, logScale=logScale, coverage=cov)
ax0.text(0.987, 0.05, minmax(datas[0][cov > 0]), transform=ax0.transAxes,  ha="right", color="k")
im0.set_cmap(cmaps[0])
divider =make_axes_locatable(ax0)
cax0=divider.append_axes("right", size="5%", pad=0.05)
cb0=plt.colorbar(im0,cax=cax0)
cb0.set_label(labels[0])
ax0.set(xlabel='X [m]',ylabel='Z [m]')
ax0.set_title(long_labels[0])
#ax0.plot(sensors[:,0],sensors[:,1], marker="x", lw=0, color="r", ms=2)
plt.show()

#1) RESISTIVITY MODEL 
logScale=True
lims={"cMin": 1000, "cMax": 30000}
ax1=plt.subplot(1,1,1)
im1=draw(ax1, mesh, datas[1], **lims, logScale=logScale, coverage=cov)
ax1.text(0.987, 0.05, minmax(datas[1][cov > 0]), transform=ax1.transAxes,  ha="right", color="k")
im1.set_cmap(cmaps[1])
divider =make_axes_locatable(ax1)
cax1=divider.append_axes("right", size="5%", pad=0.05)
cb1=plt.colorbar(im1,cax=cax1)
cb1.set_label(labels[1])
ax1.set(xlabel='X [m]',ylabel='Z [m]')
ax1.set_title(long_labels[1])
plt.show()

#2) WATER CONTENT
logScale=False
lims={"cMin": 0.0, "cMax": 0.3}
ax2=plt.subplot(1,1,1)
im2=draw(ax2, mesh, datas[2], **lims, logScale=logScale, coverage=cov)
ax2.text(0.987, 0.05, minmax(datas[2][cov > 0]), transform=ax2.transAxes,  ha="right", color="k")
im2.set_cmap(cmaps[2])
divider =make_axes_locatable(ax2)
cax2=divider.append_axes("right", size="5%", pad=0.05)
cb2=plt.colorbar(im2,cax=cax2)
cb2.set_label(labels[2])
ax2.set(xlabel='X [m]',ylabel='Z [m]')
ax2.set_title(long_labels[2])
plt.show()

#3) ICE CONTENT
logScale=False
lims={"cMin": 0.0, "cMax": 0.05}
ax3=plt.subplot(1,1,1)
im3=draw(ax3, mesh, datas[3], **lims, logScale=logScale, coverage=cov)
ax3.text(0.987, 0.05, minmax(datas[3][cov > 0]), transform=ax3.transAxes,  ha="right", color="k")
im3.set_cmap(cmaps[3])
divider =make_axes_locatable(ax3)
cax3=divider.append_axes("right", size="5%", pad=0.05)
cb3=plt.colorbar(im3,cax=cax3)
cb3.set_label(labels[3])
ax3.set(xlabel='X [m]',ylabel='Z [m]')
ax3.set_title(long_labels[3])
plt.show()

#4) AIR CONTENT
logScale=False
lims={"cMin": 0., "cMax": 0.6}
ax4=plt.subplot(1,1,1)
im4=draw(ax4, mesh, datas[4], **lims, logScale=logScale, coverage=cov)
ax4.text(0.987, 0.05, minmax(datas[4][cov > 0]), transform=ax4.transAxes,  ha="right", color="k")
im4.set_cmap(cmaps[4])
divider =make_axes_locatable(ax4)
cax4=divider.append_axes("right", size="5%", pad=0.05)
cb4=plt.colorbar(im4,cax=cax4)
cb4.set_label(labels[4])
ax4.set(xlabel='X [m]',ylabel='Z [m]')
ax4.set_title(long_labels[4])
plt.show()

#5) ROCK CONTENT
logScale=False
lims={"cMin": 0.3, "cMax": 0.8}
ax5=plt.subplot(1,1,1)
im5=draw(ax5, mesh, datas[5], **lims, logScale=logScale, coverage=cov)
ax5.text(0.987, 0.05, minmax(datas[5][cov > 0]), transform=ax5.transAxes,  ha="right", color="k")
im5.set_cmap(cmaps[5])
divider =make_axes_locatable(ax5)
cax5=divider.append_axes("right", size="5%", pad=0.05)
cb5=plt.colorbar(im5,cax=cax5)
cb5.set_label(labels[5])
ax5.set(xlabel='X [m]',ylabel='Z [m]')
ax5.set_title(long_labels[5])
plt.show()
