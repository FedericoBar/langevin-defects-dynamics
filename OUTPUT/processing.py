import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import matplotlib.mlab as mlab
from scipy.stats import norm

def maximum_ind(out_file_name):
    """
    This functions returns the maximum index for a loops
    in a simulation
    Input : out file name '/OUTPUT/out1....'
    Output : maximum index
    """
    # Fetch the data out of the out file
    file = open(out_file_name, 'r')
    data = file.readlines()
    file.close()

    INDEX = []
    for line in data[1:]:
        INDEX += [int(line.split()[1])]

    index = max(INDEX) + 1
    return index

def num_atoms(r, a):
    """
    This function computes the number of atoms containend in a loop
    of radius r (always for 111 loops since for 100 we don't need it)
    Input : radius r in Å, a lattice constant in Å
    Output : Number of atoms N
    """
    N = int(8/3*np.pi*r**3/a**3)
    if (N == 0):
        return 1
    else :
        return N

a = 3.16
N = int(sys.argv[2][6:11])
T = int(sys.argv[1])

## Start of the processing ##

dir = os.getcwd()
filename = sys.argv[2]
file = open(filename, 'r')

data_raw = file.readlines()
titles = data_raw[0].split('\t')

# Find the number of loops from data name
for long in range(12, len(filename[12:])):
    if (filename[long] == '_'):
        ind_end = long

Nloops = int(filename[14:ind_end])

# Are we in example 1 or 2 ?
example = int(filename[3])
# Change Nloops if we are in example 2:
if example == 2:
    Nloops = maximum_ind(filename)

print(Nloops)

# find dt from filename
Nlines = len(data_raw)
# Initialise the x array
time = [0]

# Create arrays that are bigger than what the output will be
# When the loop dies all the values will be 0.
position = np.zeros((Nlines, Nloops), dtype = list)
p = np.zeros((Nlines, Nloops), dtype = float)
b = np.zeros((Nlines, Nloops), dtype = list)

F_int = np.zeros((Nlines, Nloops), dtype = float)
F_BC = np.zeros((Nlines, Nloops), dtype = float)
size = np.zeros((Nlines, Nloops), dtype = float)

OmeF = [0]
dt = [1]

ind = 0

for i_line in range(len(data_raw[1:])):
    line_inter = data_raw[1:][i_line].split()

    if (float(line_inter[-2]) != time[-1]):
        time += [float(line_inter[-2])]
        OmeF += [float(line_inter[-5])]
        dt += [float(line_inter[-1])]

    # Fetch the position and burger vector in the output file
    pos = np.array([float(line_inter[2]),float(line_inter[3]),float(line_inter[4])])
    burger = np.array([float(line_inter[5]), float(line_inter[6]), float(line_inter[7])])

    position[int(line_inter[0]), int(line_inter[1])] = pos
    # Create a sensible 1D plotable array
    p[int(line_inter[0]), int(line_inter[1])] = np.dot(pos,burger)/LA.norm(burger)

    b[int(line_inter[0]), int(line_inter[1])] = burger

    # Fetch the size and the external forces in the output file
    size[int(line_inter[0]), int(line_inter[1])] = float(line_inter[10])
    #
    F_int[int(line_inter[0]), int(line_inter[1])] = float(line_inter[8])
    F_BC[int(line_inter[0]), int(line_inter[1])] = float(line_inter[9])

    ind += 1

# theoretical diffusivity in infinite medium without interaction
Diffu_init = np.zeros((Nloops,), dtype = float)
for i_loop in range(Nloops):
    if (size[0,i_loop] != 0):
        Diffu_init[i_loop] = 176.0*np.sqrt((85.0**2 + T**2)/np.abs(num_atoms(size[0,i_loop], a)))*1.0e8

Diffu_vis = np.empty((len(time) - 1, Nloops), dtype = float)


for iter in range(len(time) - 1):
    for i_loop in range(Nloops):
        Diffu_vis[iter,i_loop] = LA.norm(position[iter + 1, i_loop] - position[iter, i_loop])**2/(time[iter + 1] - time[iter])

Diffu_vis = sum(Diffu_vis, 0)
Diffu_vis = 0.5*Diffu_vis/(len(time) - 1)


plt.figure(figsize=(16,18))
plt.subplot(3,1,1)
for i in range(np.shape(p)[1]):
    plt.plot(time, p[:len(time),i], \
    label = ['Diff_init = {:.2e}'.format(round(Diffu_init[x], 0)) + \
    '... Diff = {:.2e}'.format(round(Diffu_vis[x], 0)) + \
    ' ($\AA^2$$.s^{-1}$)' for x in range(len(Diffu_init))][i])
# PLOT the limits of the box
# plt.plot(time, [np.dot(burger, np.array([1000,1000,1000])) for x in range(len(time))], linestyle = '--', color = 'r', label = 'Box limits')

# PLOT the difference between the two loops
# plt.plot(time, p[:len(time),1] - p[:len(time),0])
plt.xlabel('Time (s)')
plt.ylabel('Position along the burger axis ($\AA$)')
plt.title('Position of the loops along their burger axis')
if (Nloops < 3):
    plt.legend(loc = 'best', ncol = 2)
plt.ticklabel_format(style = 'sci', axis = 'both', scilimits = (0,0))
plt.grid()


# PLOT The external forces
# Boundary force
plt.subplot(3,1,2)
plt.xlabel('Time (s)')
plt.ylabel('Interaction force on the loops (eV/$\AA$)')
plt.plot(time, F_int[:len(time)], label = 'F_int')
plt.title('External Forces on the loops')
if (Nloops < 3):
    plt.legend(ncol = 2)
plt.ticklabel_format(style = 'sci', axis = 'both', scilimits = (0,0))
plt.grid()
# Boundary force
plt.subplot(3,1,3)
plt.plot(time, F_BC[:len(time)], label = 'F_BC')
plt.xlabel('Time (s)')
plt.ylabel('Boundary forces on the loops (eV/$\AA$)')

if (Nloops < 3):
    plt.legend(ncol = 2)
plt.ticklabel_format(style = 'sci', axis = 'both', scilimits = (0,0))
plt.grid()


# Save the figure
plt.savefig('position_' + filename[3:-4] + '.pdf', format = 'pdf')
plt.show()


#
plt.figure("Density of loops during time", figsize = (16,9))
plt.subplot(1,2,1)

N_divs = 27
section_volu = []
# z dimension in case of example 2
if example == 2:
    Nz = N/1000.0
    len_div = Nz//N_divs

    section_volu = [N**2*Nz/N_divs for x in range(N_divs)]

    section_volu = [x/(N*1e-10)**3 for x in section_volu]

else :
    len_div = N//N_divs

    for i in range(N_divs):
        section_volu = [(((i+1)*len_div)**3 - (i*len_div)**3)*1.0e-30] + section_volu

    section_volu = [x/(N*1e-10)**3 for x in section_volu]

times = [0, len(time)//2, len(time) - 1]
COLO = ['red', 'green', 'blue']
for i,t in enumerate(times):
    if example == 2:
        dist = []
        # Do not count the zeros
        for x in range(len(position[t,:])):
            if type(position[t,x]) == np.ndarray:
                dist += [position[t,x][2]]

    else :
        dist = [np.max(np.abs(position[t,x] - N/2.0)) for x in range(len(position[t, :]))]
        while N/2.0 in dist:
            dist.remove(N/2.0)

    mean = np.mean(dist)
    variance = np.var(dist)
    sigma = np.sqrt(variance)
    x = np.linspace(min(dist), max(dist), len(dist))

    weight = [section_volu[int(x%N_divs)] for x in dist]

    if example == 1:
        plt.hist(dist, np.linspace(0, N/2, N_divs), label = "t = {:.2E}s".format(time[t]), alpha = 0.5, \
                                        edgecolor = 'black', density = True, color = COLO[i], \
                                        weights = weight)
    else:
        plt.hist(dist, np.linspace(0, Nz/2, N_divs), label = "t = {:.2E}s".format(time[t]), alpha = 0.5, \
                                        edgecolor = 'black', density = True, color = COLO[i], \
                                        weights = weight)
    plt.plot(x, norm.pdf(x,mean, sigma), color = COLO[i])

plt.grid()
plt.xlabel("Distance to surface ($\AA$)")
plt.ylabel("Number of loops (Normed)")
plt.legend()

# plt.figure("Formation volume during time")
plt.subplot(1,2,2)
plt.plot(time[1:], OmeF[1:])
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Formation volume ($\AA^3$)")
plt.ticklabel_format(style = 'sci', axis = 'both', scilimits = (0,0))
# Save the figure
plt.savefig('density_' + filename[3:-4] + '.pdf', format = 'pdf')
plt.show()

plt.figure("time step during simulation")
plt.plot(time[1:], dt[1:])
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Time step (s)")
plt.ticklabel_format(style = 'sci', axis = 'both', scilimits = (0,0))
# Save the figure
plt.savefig('dt_' + filename[3:-4] + '.pdf', format = 'pdf')
plt.show()


#
