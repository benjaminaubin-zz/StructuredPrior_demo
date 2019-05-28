"""
Import libraries for plotting
"""

from matplotlib import rc, cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits import mplot3d
import matplotlib as mpl
mpl.use('TkAgg')
rc('text', usetex=True)
rc('font', family='serif')
Fontsize = 25
Fontsize_leg = 15
Linewidth = 1.5
db, do, g, cr, mv, o, dc, dg = 'dodgerblue', 'darkorange', 'gold', 'crimson', 'mediumvioletred', 'orangered', 'darkcyan', 'darkgreen'
