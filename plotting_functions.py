#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 18/12/2023
"""

'''
Functions for making plots look readable and neat
'''

import matplotlib.pyplot as plt
import numpy as np

def format_plot( ax , **kwargs ):
    '''
    Function to format plot given **kwargs
        Inputs:
            - ax: ax from "fig, ax = plt.subplots()"
            - kwargs: Formatting parameters. Mandatory parameters are as follows:
                - tick_fontsize
                - want_legend
                - legend_fontsize
                - want_grid
    '''
    ax.set_xlabel( kwargs['x_label'] , fontsize = kwargs['label_fontsize'] )
    ax.set_ylabel( kwargs['y_label'] , fontsize = kwargs['label_fontsize'] )
    ax.tick_params(axis='both', which='major', labelsize= kwargs['tick_fontsize'])
    if kwargs['want_legend'] != -1:
        leg = ax.legend(fontsize = kwargs['legend_fontsize'])
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(1)
    if kwargs['want_grid'] != -1:
        ax.grid( linestyle = "--" )
    return ax

def plot_linear_interpolation( ax, lower, upper, min_num, max_num ):
    '''
    Function to plot linear interpolation between given minimum and maximum
    '''
    ax.plot( [ min_num, max_num ],
            [ lower, upper ],
            '--',
            linewidth = 3,
            color = 'black',
            label = 'Linear interpolation',
            zorder = 0
            )
    return ax

def plot_parity_line(lower, upper):
    '''
    Function to plot parity line given lower and upper bounds of plot
    '''
    parity_line = np.linspace( lower, upper, 10 )
    fig, ax = plt.subplots()
    ax.plot(parity_line, parity_line, '--', markersize = 14, linewidth = 3, \
            color = 'lightgray', zorder = 0 )
    ax.set_aspect('equal')
    return fig, ax

def load_default_params( x_label, y_label ):
    '''
    Function to load default input parameters
        Inputs:
            - x_label, y_label: X and Y axis labels respectively
    '''
    kwargs = {'x_label': x_label,
              'y_label': y_label,
              'label_fontsize': 14,
              'want_legend': -1,
              'legend_fontsize': 12,
              'tick_fontsize': 12,
              'want_grid': -1
            }
    
    return kwargs
