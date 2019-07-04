#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

Author Federico Baraglia
Date 27.2.2019

Visualisation of the dislocation loops dynamics
Input : Position of the loops during the experience

"""

import os
import sys
import numpy as np
import time
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import product, combinations
import torch

from EOM_integrate import Np, Nz, Upot_ll, init, T
from EOM_struct import loop_obj
from useful_functions import maximum_ind

# Without this the ploting does not work..
# I don't know why
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Pytorch type and device
dtype = torch.double
device = torch.device("cpu")

# Dictionary (associates str kind to a str color)
kind_to_color = {'SIA_s' : 'blue', 'SIA_g' : 'red', 'VAC_s' : 'green', 'VAC_g' : 'yellow'}

def habit_plane(Norm):
    # This functions gives an orthogonal base of the plane containing the circle of center C and
    # normal vector N
    # Input : normal vector N:
    # Output : Orthogonal base of the plane containing the circle.

    # We find a vector orthogonal to N:
    B1 = np.cross(Norm, Norm + np.array([1,2,1]))
    # Normalisation
    B1N = B1/LA.norm(B1)
    # We find a vector orthogonal to N and B1N
    B2 = np.cross(Norm, B1N)
    # Normalisation
    B2N = B2/LA.norm(B2)
    return B1N, B2N


def plot_cir(C, r, Normal):
    # This function plots the circle of center C, radius r and normal vector N
    # Let's define a precision
    # Input : Center of the circle C, radius r, normal vector Normal and box size N:
    # Output : all the points that stand a reasonable distance from the circle
    #           representing the dislocation loop

    if (Normal[0] != 0):
        prec = 20
        eps = 0.5
        eps2 = 10
        # We start by calculating a orthogonal base of the plane containing the circle
        e1, e2 = habit_plane(Normal)

        # We define the points that are to plot:
        points = []
        # We try to do it using polar coordinates
        for thet in np.linspace(0.0, 2.0*np.pi, prec):
            for radi in np.linspace(r - eps, r + eps, prec):
                points_inter = radi * np.cos(thet) * e1 + radi * np.sin(thet) * e2

                # Is point_inter in the circle?
                a = points_inter[0]**2 + \
                    points_inter[1]**2 + \
                    points_inter[2]**2

                if (a > r**2 - eps2 and a < r**2 + eps2):
                    points += [points_inter + C]

        return points
    else :
        return []



def anima3D(Z, Np, Nz, name, TIME, Z_color):
    """
    This function saves a gif of the dislocation loops moving in the material
    Input : The loops Z, the size of the box N and the filename,
            TIME (The list containing the time step during the process)
            and Z_color containing the color associated with the type
            for every loops at every given moment of the process
    Output : animated gif showing the loop dynamics
    """
    Nt = len(Z)
    Nloops = len(Z[0])

    # Initialise the figure for the animation.
    fig_ani = plt.figure('3D dynamics of dislocation loops', figsize = (18,11))
    ax_ani = fig_ani.add_subplot(111, projection = '3d')
    ax_ani.grid(False)
    ax_ani.axis('off')

    SCT = np.empty((Nloops,), dtype = list)
    for l in range(Nloops):
        SCT[l], = ax_ani.plot([], [], [], linewidth = 0.5)

    # draw a cube
    rp = [0, Np]
    rz = [0, Nz]
    for s, e in combinations(np.array(list(product(rp, rp, rz))), 2):
        if np.sum(np.abs(s-e)) == rp[1]-rp[0] or np.sum(np.abs(s-e)) == rz[1] - rz[0]:
            ax_ani.plot3D(*zip(s, e), color="black", linewidth = 0.25)

    xs = []
    ys = []
    zs = []
    cs = []

    fps = 10
    ss = np.arange(0, Nt, 1)

    # Name of the graph and length of the axis
    title = ax_ani.text(50000, 5000, 95000, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax_ani.transAxes, ha="center")
    xdim = ax_ani.text(4000, -4500, 0, '{} $\mu$m'.format(Np//1000))
    ydim = ax_ani.text(12000, 4000, 0, '{} $\mu$m'.format(Np//1000))
    zdim = ax_ani.text(16000, 0, 8000, '{} $\mu$m'.format(Nz//1000))


    for s in ss:
        xi = []
        yi = []
        zi = []
        ci = []
        for l in range(Nloops):
            if (Z[s][l].size != 0):
                Zl_inter = plot_cir(Z[s][l].position, Z[s][l].size, Z[s][l].burger)
                xi += [[np.nan] + [x[0] for x in Zl_inter]]
                yi += [[np.nan] + [x[1] for x in Zl_inter]]
                zi += [[np.nan] + [x[2] for x in Zl_inter]]
                ci += [Z_color[s][l]]

        if (xi != [] and yi != [] and zi != []):
            xs.append(xi)
            ys.append(yi)
            zs.append(zi)
            cs.append(ci)

    # Function that updates the position of the loops during time evolution
    def update(ifrm,xa,ya,za,ca):
        num_loops = len(xa[ifrm%len(xa)])
        for l in range(num_loops):
            SCT[l].set_data(xa[ifrm%len(xa)][l], ya[ifrm%len(xa)][l])
            SCT[l].set_3d_properties(za[ifrm%len(xa)][l])
            SCT[l].set_color(cs[ifrm%len(xa)][l])
        # The loops that are out are set to [],[],[]
        for l in range(num_loops, Nloops):
            SCT[l].set_data([], [])
            SCT[l].set_3d_properties([])
        title.set_text("Number of loops = {}, Temperature = {}K, Time = {:.2E}s".format(num_loops, T, TIME[ifrm]))

    # print(xs[9], '\n', xs[10])
    # Limits for the axes
    lip = Np + Np/5
    l0p = -Np/5
    liz = 3*Nz + Nz/5
    l0z = -Nz/5
    ax_ani.set_xlim(l0p, lip)
    ax_ani.set_ylim(l0p, lip)
    ax_ani.set_zlim(l0p, lip)


    # Configure the animation
    ani = animation.FuncAnimation(fig_ani, update, Nt, fargs = (xs, ys, zs, cs), interval = 1000/fps)

    # Which repository are we in
    dir = os.getcwd()
    # Name the saved file
    fn = 'plot_' + name[3:-4]
    os.chdir(dir + '/OUTPUT/')
    # Save the file in the current directory
    ani.save(fn + '.mp4', writer='ffmpeg', fps = fps)

    # # Initialise the figure for the initial position
    # fig_still = plt.figure('3D Position of each loop')
    # ax_still = fig_still.add_subplot(111, projection = '3d')
    # sct, = ax_still.plot(xs[0], ys[0], zs[0], linewidth = 0.5)
    #
    # # Limits for the axes
    # li = N + N/5
    # ax_still.set_xlim(-N/5, li)
    # ax_still.set_ylim(-N/5, li)
    # ax_still.set_zlim(-N/5, li)
    # # Names for the axes
    # ax_still.set_xlabel('X in \AA')
    # ax_still.set_ylabel('Y in \AA')
    # ax_still.set_zlabel('Z in \AA')
    #
    # # Remove grid and axis
    # ax_still.grid(False)
    # ax_still.axis('off')
    #
    #
    # # draw cube
    # for s, e in combinations(np.array(list(product(r, r, r))), 2):
    #     if np.sum(np.abs(s-e)) == r[1]-r[0]:
    #         ax_still.plot3D(*zip(s, e), color="b", linewidth = 0.25)
    #
    # # Name of the graph
    # ax_still.set_title('Brownian motion of dislocation loops 1 ($\mu$m)')
    #
    # for angle in range(0,360):
    #     ax_still.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(0.01)

    #Name the saved file
    #fn = 'plot_3d_loops_rotate'
    #Save the file in the current directory

    return 0


"""""""""""""""""""""

\\\\\\\\\\\\\\\\\\\\\
  FUNCTIONS for Zll
\\\\\\\\\\\\\\\\\\\\\

"""""""""""""""""""""


def anima3D_Zll(name):
    """
    This function saves a gif of the dislocation loops moving in the material
    Input : output files name 'out_Nl??_FEM??_dt??.txt'
    Output : animated gif showing the loop dynamics
                and rotating animated gif (not saved...)
    """

    """
    What we are doing is converting the output file in a maner that it can be read by
    anima_3D...
    """

    # Find the number of loops
    for long in range(12, len(name[12:])):
        if (name[long] == '_'):
            ind_end = long

    # How many loops in the sample?
    Nloops = int(name[14:ind_end])
    # For example 2 (spawning defects)
    if (int(name[3]) == 2):
        Nloops = maximum_ind('/OUTPUT/' + name)


    dir = os.getcwd()
    filename = dir + '/OUTPUT/' + name
    file = open(filename, 'r')
    data = file.readlines()
    file.close()

    Nt = int(data[-1].split()[0]) + 1

    TIME = np.zeros((Nt,))
    Z = np.zeros((Nt, Nloops), dtype = loop_obj)
    Z_color = np.empty((Nt, Nloops), dtype = str)

    for i_line in range(Nt):
        for i_loop in range(Nloops):
            init_loop = loop_obj()
            init_loop.size = 0
            Z[i_line, i_loop] = init_loop

    for i_line in range(len(data[1:])):
        line = data[1:][i_line].split()
        pos = np.array([float(line[x+2]) for x in range(3)])
        bur = np.array([float(line[x+5]) for x in range(3)])

        curr_loop = loop_obj(pos, bur)

        TIME[int(line[0])] = float(line[-2])

        curr_loop.size = np.array([float(line[-4])])
        Z[int(line[0]), int(line[1])] = curr_loop

        Z_color[int(line[0]), int(line[1])] = kind_to_color[line[-3]]

    anima3D(Z, Np, Nz, name, TIME, Z_color)

    return 0

if __name__ == "__main__":
    anima3D_Zll(sys.argv[1][7:])

    # dir = os.getcwd()
    # listd = listdir(dir)
    # for name in listd:
    #     if ('out_Nl' in name):
    #         anima3D_Zll(name)
