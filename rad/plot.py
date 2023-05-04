"""
Introduction to Radar Course

Authors
=======

Zachary Chance, Robert Freking, Victoria Helus
MIT Lincoln Laboratory
Lexington, MA 02421

Distribution Statement
======================

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the United States Air Force under Air 
Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or 
recommendations expressed in this material are those of the author(s) and do not 
necessarily reflect the views of the United States Air Force.

© 2021 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. 
Government rights in this work are defined by DFARS 252.227-7013 or 
DFARS 252.227-7014 as detailed above. Use of this work other than as specifically 
authorized by the U.S. Government may violate any copyrights that exist in this work.

RAMS ID: 1016938
"""

import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pyp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import rc
import matplotlib.ticker as tck
import numpy as np

DEF_SANS = ['Segoe UI', 'Arial', 'DejaVu Sans']

def _format_ax(ax, present=False):
    
    # Check for list
    if isinstance(ax, np.ndarray):
        for axi in ax:
            _format_ax(axi)
        return

    # Color scheme
    if (present):
        face = [0, 0, 0]
        edge = [1, 1, 1]
    else:
        face = [1, 1, 1]
        edge = [0, 0, 0] 
    
    # Title
    ax.set_title(
        "", 
        size = 22, 
        name = DEF_SANS, 
        color = edge,
        weight = 'normal'
        )

    # Axis labels
    ax.set_xlabel(
        "", 
        size = 16, 
        name = DEF_SANS, 
        color = edge,
        weight = 'normal'
        )
    ax.set_ylabel(
        "", 
        size = 16, 
        name = DEF_SANS, 
        color = edge,
        weight = 'normal'
        )

    # Spines
    ax.spines['bottom'].set_color(edge)
    ax.spines['top'].set_color(edge) 
    ax.spines['right'].set_color(edge)
    ax.spines['left'].set_color(edge)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # Tick marks
    ax.tick_params(
        axis='x', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )
    ax.tick_params(
        axis='y', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )

    # Tick labels
    formatter = tck.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

def new_plot(*argv, **kwarg):
   
    axes_width = 0.4
    if 'axes_width' in kwarg.keys():
        axes_width = kwarg['axes_width']
        
    axes_height = 0.75
    if 'axes_height' in kwarg.keys():
        axes_height = kwarg['axes_height']
        
    projection = 'rectilinear'
    if 'projection' in kwarg.keys():
        projection = kwarg['projection']
        
    layout = 'center'
    if 'layout' in kwarg.keys():
        layout = kwarg['layout']
    
    # Face and edge colors
    face = [1, 1, 1]
    edge = [0, 0, 0]

    # Set font to available sans serif
    font_prop = {
        'family': 'sans-serif', 
        'sans-serif': DEF_SANS,
        'weight' : 'normal', 
        'size' : 14.0
        }
    rc(('font'), **font_prop)
    
    # Suppress font messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    # Disable open figure warning
    rc('figure', max_open_warning=0)

    # Avoid double axes
    if not (layout == 'center'):
        pyp.ioff()
    
    # Layout
    if (layout == 'center'):
        fig_height = 500
        fig_width = 1050
    elif (layout == 'sidebar'):
        fig_height = 500
        fig_width = 600
        axes_height = 0.76
        axes_width = 0.63
    elif (layout == 'sidebar-colorbar'):
        fig_height = 500
        fig_width = 700
        axes_height = 0.76
        axes_width = 0.67
    
    # Build figure
    dpi = 100
    fig = pyp.figure(
        figsize = (fig_width/dpi, fig_height/dpi), 
        dpi = dpi,
        edgecolor = edge, 
        facecolor = face
        )
    fig.canvas.header_visible = False
    ax = pyp.axes(
        [(1 - axes_width)/2, (1 - axes_height)*0.75, axes_width, axes_height],
        facecolor = face, 
        frame_on = True,
        projection = projection
        )
    
    # Avoid double axes
    pyp.ion()

    # Format axes
    _format_ax(ax)

    return fig, ax

def new_plot2(**kwarg):
   
    axes_width = 0.4
    if 'axes_width' in kwarg.keys():
        axes_width = kwarg['axes_width']
    
    present = False
    if (present):
        face = [0, 0, 0]
        edge = [1, 1, 1]
    else:
        face = [1, 1, 1]
        edge = [0, 0, 0]

    # Set font to Arial
    font_prop = {
        'family': 'sans-serif', 
        'sans-serif': DEF_SANS,
        'weight' : 'normal', 
        'size' : 14.0
        }
    rc(('font'), **font_prop)

    # Grid
    gs = {
            "left": 0.17,
            "bottom": 0.15,
            "right": 0.92,
            "top": 0.95,
            "hspace": 0.3
         }
    
    pyp.ioff()
    dpi = 100
    fig, axs = pyp.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw=gs,
        figsize = (700/dpi, 500/dpi), 
        dpi = dpi,
        edgecolor = edge, 
        facecolor = face
        )
    fig.canvas.header_visible = False
    pyp.ion()
    
    # Format axes
    _format_ax(axs)

    return fig, axs

def new_plot3(*argv, **kwarg):

    present = False
    if (present):
        face = [0, 0, 0]
        edge = [1, 1, 1]
    else:
        face = [1, 1, 1]
        edge = [0, 0, 0]

    # Set font to Arial
    font_prop = {
        'family': 'sans-serif', 
        'sans-serif': 'Arial',
        'weight' : 'bold', 
        'size' : 14.0
        }
    rc(('font'), **font_prop)

    dpi = 100
    fig = pyp.figure(
        figsize=(600/dpi, 400/dpi), 
        dpi=dpi,
        edgecolor=edge, 
        facecolor=face
        )
    ax = pyp.axes(
        [0.1667, 0.1286, 0.6667, 0.8071],
        facecolor=face, 
        frame_on=True,
        projection='3d'
        )

    # Title
    ax.set_title(
        "", 
        size = 22, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )

    # Axis labels
    ax.set_xlabel(
        "", 
        size = 16, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )
    ax.set_ylabel(
        "", 
        size = 16, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )
    ax.set_zlabel(
        "", 
        size = 16, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )

    # Spines
    ax.spines['bottom'].set_color(edge)
    ax.spines['top'].set_color(edge) 
    ax.spines['right'].set_color(edge)
    ax.spines['left'].set_color(edge)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # Tick marks
    ax.tick_params(
        axis='x', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )
    ax.tick_params(
        axis='y', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )
    ax.tick_params(
        axis='z', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )

    # Tick labels
    formatter = tck.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)

    # Turn on interaction
    pyp.ion()
    
    return fig, ax

def new_plot_text(*argv, **kwarg):

    present = False
    if (present):
        face = [0, 0, 0]
        edge = [1, 1, 1]
    else:
        face = [1, 1, 1]
        edge = [0, 0, 0]

    # Set font to Arial
    font_prop = {
        'family': 'sans-serif', 
        'sans-serif': 'Arial',
        'weight' : 'bold', 
        'size' : 14.0
        }
    rc(('font'), **font_prop)

    dpi = 100
    fig = pyp.figure(
        figsize = (800/dpi, 400/dpi), 
        dpi = dpi,
        edgecolor = edge, 
        facecolor = face,
        title=''
        )
    ax = pyp.axes(
        [0.1667, 0.1286, 0.5, 0.8071],
        facecolor = face, 
        frame_on = True
        )

    # Title
    ax.set_title(
        "", 
        size = 22, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )

    # Axis labels
    ax.set_xlabel(
        "", 
        size = 16, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )
    ax.set_ylabel(
        "", 
        size = 16, 
        name = 'Arial', 
        color = edge,
        weight = 'bold'
        )

    # Spines
    ax.spines['bottom'].set_color(edge)
    ax.spines['top'].set_color(edge) 
    ax.spines['right'].set_color(edge)
    ax.spines['left'].set_color(edge)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # Tick marks
    ax.tick_params(
        axis='x', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )
    ax.tick_params(
        axis='y', 
        colors = edge, 
        labelsize = 14.0, 
        width = 2.0
        )

    # Tick labels
    formatter = tck.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Turn on interaction
    pyp.ion()
    
    return fig, ax

def new_rad_plot(*argv, **kwarg):

    present = False
    if (present):
        face = [0, 0, 0]
        edge = [1, 1, 1]
    else:
        face = [1, 1, 1]
        edge = [0, 0, 0]

    # Set font to Arial
    font_prop = {
        'family': 'sans-serif', 
        'sans-serif': DEF_SANS,
        'weight' : 'normal', 
        'size' : 14.0
        }
    rc(('font'), **font_prop)

    pyp.ioff()
    
    dpi = 100
    fig_scan = pyp.figure(
        figsize = (9*90/dpi, 7*90/dpi), 
        dpi = dpi,
        edgecolor = edge, 
        facecolor = face
        )
    fig_scan.canvas.header_visible = False
    ax_scan = fig_scan.add_axes(
        [0.05, 0.025, 0.93, 0.93],
        facecolor=face, 
        frame_on=True,
        projection='polar'
        )
    fig_pulse = pyp.figure(
        figsize = (9*90/dpi, 2*110/dpi), 
        dpi = dpi,
        edgecolor = edge, 
        facecolor = face
        )
    fig_pulse.canvas.header_visible = False
    ax_pulse = fig_pulse.add_axes(
        [0.1, 0.27, 0.85, 0.7],
        facecolor=face, 
        frame_on=True,
        projection='rectilinear'
        )

    pyp.ion()
    
    # Put zero azimuth at top
    ax_scan.set_theta_zero_location("N")
    ax_scan.set_theta_direction(-1)
    
    # Title
    ax_scan.set_title(
        "", 
        size = 22, 
        name = DEF_SANS, 
        color = edge,
        weight = 'normal'
        )
    
    # Turn on interaction
    pyp.ion()
    
    return fig_scan, ax_scan, fig_pulse, ax_pulse