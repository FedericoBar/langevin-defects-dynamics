# -*- coding: utf-8 -*-
"""
Author Federico Baraglia
Date 08.02.2020

Resolution of overdamped Langevin Equation of Motion


Output: Position of the dislocation loops during time

"""

# Let's define the unit system
# L  T  E   F     cont
# Å  s  eV  eV/Å  eV/Å3

# Packages needed
import os
import sys
import torch
import copy
import time
import numpy as np
import pickle
import subprocess
# import numpy.linalg as LA

# Homemade scripts
import useful_functions as uf
import boundaryForce as BF
import linked
import parallel
import writing
import EOM_struct as EOMs
import set_init


"""
Definition of some constants useful throughout the resolution
"""
# Physics constants
kB = 8.617e-5 # Boltzmann's constant eV/K

# Workflow parameters
# Finite elements or no ?
FEA = 1
# Which example do we wan't to simulate?
# 1 corresponds to a box (10 microns ** 3) with a initial set of defects
# 2 corresponds to a thin film (10 microns ** 3 * 1 micron) with spawning defects
# according to DFT data.
example = 1
# Where is the DFT data from Daniel?
data_ex3 = 'test.cascade.110.dat'

# Mechanical properties
# Lamé coefficients for W Tungsten
nu = 0.278
mu = 1.006 # eV/Å3
E = 2.56875 # eV/Å3 Young modulus

# Integration constants
dzi = 1.0e-2 # We define the space step in the direction of the movement of the loop
Nt = 2000 # Number of time steps

# Geometric properties
# Thin film:
Np = 10000 # Size 1 of the cell (in Å)
if example == 1:
    Nz = 10000 # Size 2 of the sample (in Å)
else :
    Nz = 1000 # Size 2 of the sample (in Å)

Nm = 14 # Number of element per side
# length of division per side+
Nbr = 1000
dim_Zllp = Np//Nbr
dim_Zllz = Nz//Nbr

# The lattice constant is defined in EOM_struct.py

# Adaptive time step config
ATS = True
dt_init = 1.0e-6 # Initial time step
dt = dt_init # current time step s
dt_min = 1e-10 # Minimum time step
dt_max = 1e-01 # Maximum time step
ratio_dt_min = 0.5 # Minimum change in time step
ratio_dt_max = 1.5 # Maximum change in time step
DELt = 20 # Number of time steps with constant step.
max_disp = Np/25 # or 200 in Angstroms
min_disp = Np/25 # or 80 in Angstroms

# Physical condition of the problem
T = 300 # Temperature K
Nloops = 500 # Number of loops
ion_flux = 6.25e14 * 1.0e-20 # Ion flux (W+/Å**2/s)

"""
Definition of some parameters for CUDA/CPU management
"""
# Float type for tensor definition
dtype = torch.double
# Initial device for tensor definition
device = torch.device('cpu')
# Number of GPU devices we want:
nbr_devices = 0
# Are we CPU parralel  ?
CPU_par = True
# Number of CPU for UPOT
Ncores_dzu = 20

# Number of CPU for stressBC
if (Np == Nz):
    Ncores_BC = 20
else :
    Ncores_BC = 20

# Number of CPU for Cast3m processing.dgibi
Ncores_FE = 20

"""
END DEFINITION OF CONSTANTS
"""


def energy_int(r, Pa, Pb):
    """
    This function compute the energy of interaction between two dislocation loops
    Input : dipole tensor Pa[np.array 3*3], Pb[np.array 3*3], r (vector pointing from a to b)
    Output : interaction energy between loop a and b
    For more details see Eq(2) in PRM 2, 033602
    """
    # curr_devi = r.device

    # If r is 0 it means we are considering two identical loops.
    # the energy of interaction must therefore be 0
    # The equal method must be used on tensors saved in the same device (hence
    # device = curr_devi)
    if (r.equal(torch.tensor([0, 0, 0], dtype = dtype, device = device))):
        return 0
    else:

        # G_ijkl = torch.tensor(uf.greenDD_mod(r,nu,mu).reshape(3,3,3,3), dtype = dtype).to(curr_devi)
        G_ijkl = torch.tensor(uf.greenDD_mod(r,nu,mu).reshape(3,3,3,3), dtype = dtype)

        # # According to linear elasticity:
        # #    E_interaction = P^a_ij * G_ik,jl * P^b_kl
        # # The tensordot function must be used for tensor saved on the same device
        # # (Hence device = curr_devi ...)

        E_int = torch.tensordot(Pa, torch.tensordot(G_ijkl, Pb, 2))

        return E_int


def Upot_ll(loop, cell, replica_Zll, reduced_Zll, R_cut = dim_Zllp - 1):
    """
    This function computes the elastic interaction potential between the loop
    considered and all the other loop (the other interaction will vanish due
    to finit difference)
    Compute the interaction potential for N loops ( U = sum_i,z EInt_i,z )
    appropriate with linked list structure
    Input : loop node, cell it is in, Replicas of Zll on the GPU's, cut off radius
    Output : interaction energy (pair wise interaction) (only between loop and all
    the other loops)
    """
    # t_s = 0
    # t_s_neigh = 0

    # Interaction energy computation
    U_inter = 0
    NEIGHBOURS = linked.neighbours(cell, dim_Zllp, dim_Zllz, R_cut)

    # We run through all the cells of the system.
    #               E_interaction = left (a) * Green * right (b)
    for i in range(dim_Zllp):
        for j in range(dim_Zllp):
            for k in range(dim_Zllz):
                curr_list = replica_Zll[i,j,k]

                curr_node = curr_list.head
                # Run through the nodes of curr_lists[2*i_device]
                while (curr_node != None):
                    if ([i,j,k] in NEIGHBOURS):
                        r = curr_node.val.position - loop.val.position
                        dta = loop.val.dtensor
                        dtb = curr_node.val.dtensor

                        U_inter += energy_int(r, dta, dtb)
                    else :
                        r = reduced_Zll[i,j,k].position - reduced_Zll[cell[0], cell[1], cell[2]].position
                        dta = reduced_Zll[cell[0], cell[1], cell[2]].dtensor
                        dtb = reduced_Zll[i,j,k].dtensor

                        U_inter += energy_int(r, dta, dtb)

                    # Go to next node of curr_lists[2*i_device]
                    curr_node = copy.deepcopy(curr_node.next)

    # Return the interaction energy (sum of all the pbs on each GPU divided by two ((pair interaction)) )
    # print('Proportion of time in neighbours : ', t_s_neigh/(t_s+t_s_neigh))
    # print('Proportion of time not in neighbours : ', t_s/(t_s+t_s_neigh))

    return U_inter

def Upot_ll_GPU(replicas_Zll, R = dim_Zllp - 1):
    """
    This function computes the interaction potential between all the loops
    Compute the interaction potential for N loops ( U = 1/2 sum_i,j EInt_i,j )
    appropriate with linked list structure
    Input : Replicas of Zll on the GPU's and cut off radius
    Output : interaction energy (pair wise interaction)
    """
    # I have to change something in order for it to be really GPU //
    # Initialise some empty structure for avoiding conflict between tensors on different
    # GPU's:
        #
    curr_lists = [EOMs.linklist() for i in range(2*nbr_devices)]
        #
    curr_nodes = [EOMs.node() for i in range(2*nbr_devices)]
        # Interaction energy computation is divided on the number of GPU availables
    U_inters = [0 for i in range(nbr_devices)]

        # vector separating two loops
    r = np.empty((nbr_devices,), dtype = list)
        # 1st dipole tensor (loop a)
    dta = np.empty((nbr_devices,), dtype = list)
        # 2nd dipole tensor (loop b)
    dtb = np.empty((nbr_devices,), dtype = list)

    # Let's compute all the reduced loops of the grid :


    """
    /!\
    WARNING : i_device can be i only because we have same nbr of GPU and cell per side....
    for now it is suitable but we have to be careful if we are to change the division on
    the GPU's
    /!\
    """
    # We run through all the cells of the system. Dividing all the cell in groups containing
    # an equal number of cell. Each group will be treated by an individual GPU.
    # In the present case, we divide the cube in slice i == cste.

    # the even i_device will be for the left term in the interaction energy
    # whereas the odd i_device will be for the right term :
    #               E_interaction = left (a) * Green * right (b)
    for i_device in range(nbr_devices):
        # Run through the cell of the group i == i_device
        for j in range(dim_Zllp):
            for k in range(dim_Zllz):
                # Who is neighbour to i,j,k :
                NEIGHBOURS = linked.neighbours([i_device,j,k], dim_Zll, R)
                # curr_lists[2*i_device] is the linked list of the cell i_device,j,k
                curr_lists[2*i_device] = replicas_Zll[i_device][i_device,j,k]
                # curr_nodes[2*i_device] will run through the nodes of curr_lists[i_device]
                curr_nodes[2*i_device] = curr_lists[2*i_device].head

                # Run through the nodes of curr_lists[2*i_device]
                while (curr_nodes[2*i_device] != None):
                    # We compute the interaction energy with all the nodes
                    for ii in range(dim_Zllp):
                        for jj in range(dim_Zllp):
                            for kk in range(dim_Zllz):
                                # Are we a neighbour ?
                                if ([ii,jj,kk] in NEIGHBOURS):
                                    # curr_lists[2*i_device + 1] is the linked list of the cell ii, jj, kk
                                    curr_lists[2*i_device + 1] = replicas_Zll[i_device][ii,jj,kk]
                                    # curr_nodes[2*i_device + 1] will run through the nodes of curr_lists[2*i_device + 1]
                                    curr_nodes[2*i_device + 1] = curr_lists[2*i_device + 1].head

                                    # Run through the nodes of curr_lists[2*i_device + 1]
                                    while (curr_nodes[2*i_device + 1] != None):
                                        # Fetch the vector separating a from b, and the two dipole tensors
                                        r[i_device] = curr_nodes[2*i_device + 1].val.position - curr_nodes[2*i_device].val.position
                                        dta[i_device] = curr_nodes[2*i_device].val.dtensor
                                        dtb[i_device] = curr_nodes[2*i_device + 1].val.dtensor

                                        # Compute the interaction energy.
                                        U_inters[i_device] += energy_int(r[i_device], dta[i_device], dtb[i_device])

                                        # Go to the next node of curr_lists[2*i_device + 1]
                                        curr_nodes[2*i_device + 1] = copy.deepcopy(curr_nodes[2*i_device + 1].next)

                                else:
                                    if (replicas_Zll[i_device][ii,jj,kk].head != None):
                                        # Reduced loop:
                                        reduced_loop = replicas_Zll[i_device][ii,jj,kk].redu(i_device)

                                        r[i_device] = reduced_loop.position - curr_nodes[2*i_device].val.position
                                        dta[i_device] = curr_nodes[2*i_device].val.dtensor
                                        dtb[i_device] = reduced_loop.dtensor

                                        U_inters[i_device] += energy_int(r[i_device], dta[i_device], dtb[i_device])

                    # Go to next node of curr_lists[2*i_device]
                    curr_nodes[2*i_device] = copy.deepcopy(curr_nodes[2*i_device].next)

    #Wait for all the operations on the devices to terminate.
    for i_device in range(nbr_devices):
        # Set device to a working device and synchronize it
        torch.cuda.synchronize('cuda:' + str(i_device))

    for i_device in range(nbr_devices):
        # If the value of U_inter[i_device] was unchanged (because no loops are on the
        # section of the problem on the GPU i_device)
        # We set the value to a tensor 0 on the CPU
        if (U_inters[i_device] == 0):
            U_inters[i_device] = torch.tensor([0], dtype = dtype, device = device)
        # If the value was changed during the process, then we only send the tensor
        # to the CPU
        else:
            U_inters[i_device] = U_inters[i_device].to(device)

    # return (U_inter0.to('cpu') + U_inter1.to('cpu') + U_inter2.to('cpu') + U_inter3.to('cpu'))/2

    # Return the interaction energy (sum of all the pbs on each GPU divided by two ((pair interaction)) )
    return 0.5*sum(U_inters)


def DzU_ll(Zll, reduced_Zll, node_interest, cell_id, ind):
    """
    This function compute the derivative of U with respect to zi
    (position of th ith loop's centre of mass)
    This is the function appropriate for working with linked list
    Input : replicas of Zll on the GPUs,
            node where we want the gradient of U,
    		cell coordinates [i, j, k], place of the node in the linked list ind
    Output : gradient of the potential of interaction in zi
    """
    # i, j, k are the coordinates of the cell where the loop considered is in
    i = cell_id[0]
    j = cell_id[1]
    k = cell_id[2]

    # Number of GPU devices we are using
    # nbr_devices = len(replicas_Zll)

    # corresponding dx, dy and dz
    dr = torch.tensor([dzi*x for x in node_interest.val.burger], dtype = dtype)
    # position of the considered loop
    r = node_interest.val.position

    # We are going to change slightly the position of one node of the system
    # Therefore, we need to replace the current linked list containing the node
    # we want to move with a new list (for each GPU)
    # Initialise the linked lists of the replicas

    curr_list = Zll[i,j,k]
    curr_node = curr_list.head
    for noeud in range(ind):
        curr_node = curr_node.next
    # Compute the value of the interaction energy when the considered node is Slightly
    # moved on the positive side (change the position on all the replicas)
    curr_node.val.position = (r + dr)
    if Nloops > 30:
        Rcut = 1
    else :
        Rcut = dim_Zllp - 1

    value_p = Upot_ll(curr_node, [i,j,k], Zll, reduced_Zll, Rcut)

    # Compute the value of the interaction energy when the considered node is Slightly
    # moved on the negative side (change the position on all the replicas)
    curr_node.val.position = (r - dr)
    # Cut_off radius = 1
    value_m = Upot_ll(curr_node, [i,j,k], Zll, reduced_Zll, Rcut)

    # Reset the position to the initial one
    curr_node.val.position = r
    return ((value_p - value_m)/(2*dzi))


def GradU_ll(Zll, node_interest, cell_id, ind):
    """
    This function compute the gradient of U
    This is the function appropriate for working with linked list
    Input : replicas of Zll on the GPUs,
            node where we want the gradient of U,
    		cell coordinates [i, j, k], place of the node in the linked list ind
    Output : gradient of the potential of interaction in zi
    """
    # i, j, k are the coordinates of the cell where the loop considered is in
    i = cell_id[0]
    j = cell_id[1]
    k = cell_id[2]

    # Number of GPU devices we are using
    # nbr_devices = len(replicas_Zll)

    # corresponding dx, dy and dz
    dx = torch.tensor([dzi, 0, 0], dtype = dtype)
    dy = torch.tensor([0, dzi, 0], dtype = dtype)
    dz = torch.tensor([0, 0, dzi], dtype = dtype)
    # position of the considered loop
    r = node_interest.val.position

    # Initialise the gradient
    GradU = torch.zeros(3,3, dtype = dtype)

    for i in range(3):
        # We are going to change slightly the position of one node of the system
        # Therefore, we need to replace the current linked list containing the node
        # we want to move with a new list (for each GPU)
        # Initialise the linked lists of the replicas

        curr_list = Zll[i,j,k]
        curr_node = curr_list.head
        for noeud in range(ind):
            curr_node = curr_node.next

        # Compute the value of the interaction energy when the considered node is Slightly
        # moved on the positive side (change the position on all the replicas)
        curr_node.val.position = (r + [dx, dy, dz][i])
        # Cut_off radius = 1
        value_p = Upot_ll(Zll, 1)
        # Compute the value of the interaction energy when the considered node is Slightly
        # moved on the negative side (change the position on all the replicas)
        curr_node.val.position = (r - [dx, dy, dz][i])
        # Cut_off radius = 1
        value_m = Upot_ll(Zll, 1)

        # Reset the position to the initial one
        curr_node.val.position = r

        GradU[i] = ((value_p - value_m)/(2*dzi))


    return GradU

def DzU_ll_GPU(replicas_Zll, node_interest, cell_id, ind):
    """
    This function compute the derivative of U with respect to zi
    (position of th ith loop's centre of mass)
    This is the function appropriate for working with linked list
    Input : replicas of Zll on the GPUs,
            node where we want the gradient of U,
    		cell coordinates [i, j, k], place of the node in the linked list ind
    Output : gradient of the potential of interaction in zi
    """
    # i, j, k are the coordinates of the cell where the loop considered is in
    i = cell_id[0]
    j = cell_id[1]
    k = cell_id[2]

    # Number of GPU devices we are using
    nbr_devices = len(replicas_Zll)

    # corresponding dx, dy and dz
    dr = torch.tensor([dzi*x for x in node_interest.val.burger], dtype = dtype)
    # position of the considered loop
    r = node_interest.val.position

    # We are going to change slightly the position of one node of the system
    # Therefore, we need to replace the current linked list containing the node
    # we want to move with a new list (for each GPU)
    # Initialise the linked lists of the replicas
    curr_lists = [EOMs.linklist() for i in range(nbr_devices)]
    curr_nodes = [EOMs.node() for i in range(nbr_devices)]

    for i_device in range(nbr_devices):
        curr_lists[i_device] = replicas_Zll[i_device][i,j,k]
        curr_nodes[i_device] = curr_lists[i_device].head
        for noeud in range(ind):
            curr_nodes[i_device] = curr_nodes[i_device].next


    # Compute the value of the interaction energy when the considered node is Slightly
    # moved on the positive side (change the position on all the replicas)
    for i_device in range(nbr_devices):
        curr_devi = torch.device('cuda:' + str(i_device))
        curr_nodes[i_device].val.position = (r + dr).to(curr_devi)
    value_p = Upot_ll_GPU(replicas_Zll, 1)
    # Compute the value of the interaction energy when the considered node is Slightly
    # moved on the negative side (change the position on all the replicas)
    for i_device in range(nbr_devices):
        curr_devi = torch.device('cuda:' + str(i_device))
        curr_nodes[i_device].val.position = (r - dr).to(curr_devi)
    value_m = Upot_ll_GPU(replicas_Zll, 1)

    # Reset the position to the initial one (all the devices)
    for i_device in range(nbr_devices):
        curr_devi = torch.device('cuda:' + str(i_device))
        curr_nodes[i_device].val.position = r.to(curr_devi)

    return ((value_p - value_m)/(2*dzi))


def GradU_ll_GPU(replicas_Zll, node_interest, cell_id, ind):
    """
    This function compute the gradient of U
    This is the function appropriate for working with linked list
    Input : replicas of Zll on the GPUs,
            node where we want the gradient of U,
    		cell coordinates [i, j, k], place of the node in the linked list ind
    Output : gradient of the potential of interaction in zi
    """
    # i, j, k are the coordinates of the cell where the loop considered is in
    i = cell_id[0]
    j = cell_id[1]
    k = cell_id[2]

    # Number of GPU devices we are using
    nbr_devices = len(replicas_Zll)

    # corresponding dx, dy and dz
    dr = torch.tensor([dzi*x for x in node_interest.val.burger], dtype = dtype)
    # position of the considered loop
    r = node_interest.val.position

    # We are going to change slightly the position of one node of the system
    # Therefore, we need to replace the current linked list containing the node
    # we want to move with a new list (for each GPU)
    # Initialise the linked lists of the replicas
    curr_lists = [EOMs.linklist() for i in range(nbr_devices)]
    curr_nodes = [EOMs.node() for i in range(nbr_devices)]

    for i_device in range(nbr_devices):
        curr_lists[i_device] = replicas_Zll[i_device][i,j,k]
        curr_nodes[i_device] = curr_lists[i_device].head
        for noeud in range(ind):
            curr_nodes[i_device] = curr_nodes[i_device].next

    GradU = torch.zeros(3, dtype = dtype)
    for i in range(3):
        # Compute the value of the interaction energy when the considered node is Slightly
        # moved on the positive side (change the position on all the replicas)
        for i_device in range(nbr_devices):
            curr_devi = torch.device('cuda:' + str(i_device))
            curr_nodes[i_device].val.position = (r + dr).to(curr_devi)
        value_p = Upot_ll_GPU(replicas_Zll, 1)
        # Compute the value of the interaction energy when the considered node is Slightly
        # moved on the negative side (change the position on all the replicas)
        for i_device in range(nbr_devices):
            curr_devi = torch.device('cuda:' + str(i_device))
            curr_nodes[i_device].val.position = (r - dr).to(curr_devi)
        value_m = Upot_ll_GPU(replicas_Zll, 1)

        # Reset the position to the initial one (all the devices)
        for i_device in range(nbr_devices):
            curr_devi = torch.device('cuda:' + str(i_device))
            curr_nodes[i_device].val.position = r.to(curr_devi)

        GradU[i] = (value_p - value_m)/(2*dzi)

    return GradU


def init(Nloops, FEA):
    """
    This function initialize all the parameters for the Langevin EOM
    Input : Number of loops, FEA (if we are considering finite element or not)
    Output : Returns the linked list system and the list of all the cascades caracs
        extracted from Daniel's DFT script
    """


    # Initialise the loops (with N loops)
    Z = []

    # Possible degenerate direction:
    degenerate_direction = [torch.tensor([1,1,1]), torch.tensor([1,1,-1])\
                            , torch.tensor([1,-1,1]), torch.tensor([-1,1,1])]

    # Possible shape and nature possible
    # nature_poss = ['SIA', 'VAC_open', 'VAC_close']
    # shape_poss = ['Cir']
    # Possible Burger vector:
    burger_poss = [0.5 * torch.tensor([1,1,1], dtype = dtype, device = device)]*9 + \
                   [torch.tensor([1,0,0], dtype = dtype, device = device),\
                   torch.tensor([0,1,0], dtype = dtype, device = device),\
                   torch.tensor([0,0,1], dtype = dtype, device = device)]


    #nature = [nature_poss[0]] * Nloops
    #shape = [shape_poss[0]] * Nloops
    burger = [burger_poss[np.random.randint(0, 12)] for x in range(Nloops)]
    # burger = [burger_poss[np.random.randint(0, 1)] for x in range(Nloops)]
    radius = torch.tensor([np.random.randint(40, 100) for x in range(Nloops)])
    # radius = torch.tensor([40])
    # pos = [[250, 250, 250], [7000, 7000, 7000]]

    # initialise all the loops :
    for ind_loops in range (Nloops):
        curr_burger = copy.deepcopy(burger[ind_loops])
        # Compute the dipole tensor coordinates accordingly
        # We won't access it everytime...
        # os.system('python dipoleTensor.py ' + nature[ind_loops] + ' ' + \
        # str(radius[ind_loops]) + ' ' + shape[ind_loops] + ' ' + str(length) + ' ' + str(B))
        # P = uf.fetchComponents(str(B))

        # Use of the analytical solution in order to compute the dipole tensor coordinates
        P = uf.dipole_coord_analytical(curr_burger, radius[ind_loops], nu, mu, EOMs.a)

        # Random degenerate direction
        deg_dir = np.random.randint(0, 4)
        # deg_dir = np.random.randint(0, 1)
        # Change the dipole tensor coordinates and burger vector
        if (curr_burger.equal(torch.tensor([1,0,0], dtype = dtype)) \
            or curr_burger.equal(torch.tensor([0,1,0], dtype = dtype)) \
            or curr_burger.equal(torch.tensor([0,0,1], dtype = dtype))):
            pass
        else:
            P, curr_burger = uf.degenerate_loop(P, curr_burger, degenerate_direction[deg_dir])
        # Is the loop a self interstitial (+1) or a vacancy loop  (-1)?
        vac_SIA = np.random.choice([1,1,1,-1])
        # vac_SIA = 1
        radius[ind_loops] = uf.sia_or_vac(radius[ind_loops], vac_SIA)

        # Random starting position
        start_position = torch.randint(0, Np, (3,), dtype = dtype, device = device)
        # The z coordinate must be multiplied by Nz/Np
        start_position[2] *= Nz/Np

        # start_position = torch.tensor(pos[ind_loops], dtype = dtype)
        start_loop = EOMs.loop_obj(start_position, curr_burger, P, radius[ind_loops])
        Z = Z + [start_loop]

    # Create the linked list system
    Zll = linked.gridalize(Z, Np, Nz, Nbr)

    # Create the output file for
        # 1. output
    writing.init_write(example, Np, Nloops, T, FEA, dt)
        # 2. Parameters
    writing.write_param(Np, Nz, Nm, Nloops, nu, E)

    # Extract DFT data
    if (example == 2):
        loops_data = set_init.extract_loop_data(data_ex3)
    else :
        loops_data = None

    return Zll, loops_data


        ####################################
        ###  Main part of the program :  ###
        ####################################


def main(dt, FEA = 1):
    """
    This function integrate the Langevin EOM
    Input : FEA = bool (True if we consider the force acting on the loops due to the
    boundary False if not)
    Output : Position of the center of mass for each loops
    """
    # This list will contain the time steps during the simulation
    DT = [dt]

    ETA = 0 # Initialise time s

    print('Start initialisation...')
    Zll, loops_data = init(Nloops, FEA)
    print('End initialisation.')
    # What is the maximum number of loops ?
    Nmax = Nloops

    # Formation volume associated with the initial configuration
    for_volume = uf.formation_volume(Zll, nu, mu, EOMs.a)

    # Write in the out file the initial state
    F_init = torch.zeros((Nmax,), dtype = dtype)
    writing.write_pos_ll(Zll, 0, example, Np, Nloops, T, FEA, dt, F_init, F_init, ETA, for_volume)

    # Show the initial state
    for i in range(dim_Zllp):
        for j in range(dim_Zllp):
            for k in range(dim_Zllz):
                print(i,j,k)
                Zll[i,j,k].show()

    print('\n\n')

    # DZI will contain the average displacement for each loop over %DELt% iterations
    DZI = np.zeros((Nmax, ))

    # We use a basic Euler-Maruyama numerical method for solving the SDE
    #
    #                   w(t_n+1) = w(t_n) + dt*(-D/kbT grad(E).b/|b| + sqrt(2D) xi)
    #
    for time_iter in range(Nt - 1):
        # Change the time step according to the maximum displacement during the period
        # every DELt time steps
        if (ATS and time_iter > 0 and time_iter%DELt == 0):
            # We compute the average displacement for all the loops during DELt steps
            DZI = DZI/DELt
            avg_disp = max(DZI)
            # If the max average displacement is higher (lower) than a value,
            # dt is increased (decreased) respectively
            if (avg_disp > max_disp):
                dt *= ratio_dt_min
            if (avg_disp <= min_disp):
                dt *= ratio_dt_max
            # dt Cannot exit the bounds fixed at the beginning. This is to avoid
            # Exotic time step...
            if (dt > dt_max) :
                dt = dt_max
            if (dt < dt_min) :
                dt = dt_min
            # Add it to the list of time steps.
            DT += [dt]

            # Return to zeros the average displacement of the loops
            DZI = np.zeros((Nmax, ))

        t0_ti = time.time()
        # Create system of reduced defects
        reduced_Zll = np.empty((dim_Zllp, dim_Zllp, dim_Zllz), dtype = EOMs.loop_obj)

        for i in range(dim_Zllp):
            for j in range(dim_Zllp):
                for k in range(dim_Zllz):
                    reduced_Zll[i,j,k] = Zll[i,j,k].redu()
        # Pickle the reduced Zll
        f = open('rzll.pckl', 'wb')
        pickle.dump(reduced_Zll, f)
        f.close()
        # Pickle Zll
        f = open('zll.pckl', 'wb')
        pickle.dump(Zll, f)
        f.close()

        # Initialise the forces acting on the loops
        # (This is for output)
        # Force from surface to the loops
        F_BC = np.zeros((Nmax, ), dtype = float)
        # Force from interaction between the loops
        F_INT = np.zeros((Nmax, ), dtype = float)

        # For communication with Cast3m
            # For this we create a textfile containing:
                # Position X, Y, Z
                # Burger vector
                # Dipole tensor
        writing.write_Zll(Zll, time_iter)

        # Initialise the forces due to the nearby surface
        force_boundary_l = np.zeros((Nmax,))
        force_boundary_l_3d = np.zeros((Nmax,), dtype = list)

        """
        Below is the FEA to compute the force acting on the loop due to the boundary
        """
        if (FEA == 1):
            t_FEA = time.time()
            # Let's run the Finite element analysis to find the force on each defect
            # due to the nearby surface.
            print('Start Finite Element ...')
            subprocess.run('python runFEM.py {} {}'.format(time_iter, CPU_par), shell = True)
            print('End Finite Element. Time : {}s\n'.format(time.time() - t_FEA))

            t_BC = time.time()

            print('Start boundary Force computation...')
            for i in range(dim_Zllp):
                for j in range(dim_Zllp):
                    for k in range(dim_Zllz):
                        # curr_list is the linked list associated with cell i,j,k
                        curr_list = Zll[i,j,k]
                        # curr_node will run through the nodes of curr_list
                        curr_node = curr_list.head

                        # This new list will contain the linked list of the forces applied
                        # to curr_node in the cell i,j,k
                        new_list = EOMs.linklist()
                        # Run through the nodes of curr_list
                        while (curr_node != None):
                            if not(torch.abs(curr_node.val.burger).equal(torch.tensor([1,0,0], dtype = dtype)) \
                                    or torch.abs(curr_node.val.burger).equal(torch.tensor([0,1,0], dtype = dtype)) \
                                    or torch.abs(curr_node.val.burger).equal(torch.tensor([0,0,1], dtype = dtype))):
                                if not(-1.556 < curr_node.val.size < -1.555):
                                    # Create the node structute which value is the Boundary force on curr_node
                                    force_boundary_l[curr_node.id] = BF.boundaryForce(curr_node.id)
                                else :
                                    force_boundary_l_3d[curr_node.id] = BF.boundaryForce3d(curr_node.id)
                            else:
                                force_boundary_l[curr_node.id] = 0
                                force_boundary_l_3d[curr_node.id] = 0
                            # Go to the next node
                            curr_node = curr_node.next

            print('End boundary Force computation. Time : {}s'.format(time.time() - t_BC))

        """
        End of FEA section
        """
        Zll_timeIter = copy.deepcopy(Zll)
        if nbr_devices > 0:
            replicas_Zll = parallel.broadcast_system(Zll_timeIter, nbr_devices)

        if (CPU_par == True):
            # We compute elastic interaction energy derivative with mpi4py
            subprocess.run('mpiexec -n {} python mpi_dzu.py'.format(Ncores_dzu), shell = True)
            # fetch the result:
            f = open('dzu.pckl', 'rb')
            DzU = pickle.load(f)
            f.close()

        # Let's compute the new position for each node (contained in all the
        # linked lists systems)
        for i in range(dim_Zllp):
            for j in range(dim_Zllp):
                for k in range(dim_Zllz):
                    # Curr list is the linked list of cell [i,j,k]
                    curr_list = Zll[i,j,k]
                    # Curr_node will run through the nodes of curr_list
                    curr_node = curr_list.head

                    # ind will be the position of the loop on the linked list
                    ind = 0
                    # curr_node.id could do the job aswell...
                    while (curr_node != None):
                        t1 = time.time()
                        # Force applied on the dislocation loops
                        # F = F_interaction + F_boundary
                        # We compute the interaction energy only if the loop is glissile
                        if not(torch.abs(curr_node.val.burger).equal(torch.tensor([1,0,0], dtype = dtype)) \
                                or torch.abs(curr_node.val.burger).equal(torch.tensor([0,1,0], dtype = dtype)) \
                                or torch.abs(curr_node.val.burger).equal(torch.tensor([0,0,1], dtype = dtype))):
                            # If we are not a VAC defect or a void
                            # Then we compute the interaction force as usual with the directional derivative
                            if not(-5 <= curr_node.val.size <= -1.5559):
                                if (CPU_par == True):
                                    force_interaction = DzU[curr_node.id]
                                elif (nbr_devices > 0):
                                    force_interaction = DzU_ll_GPU(replicas_Zll, curr_node, [i,j,k], ind)
                                else :
                                    force_interaction = DzU_ll(Zll_timeIter, reduced_Zll, curr_node, [i,j,k], ind)
                            # If the defect is a VAC or a VOID, we compute the elastic interaction force
                            # using the gradient function.
                            # However, since we consider the VAC and VOIDs as being sessile
                            else :
                                # if (nbr_devices > 0):
                                #     force_interaction = GradU_ll_GPU(replicas_Zll, curr_node, [i,j,k], ind)
                                # else:
                                #     force_interaction = GradU_ll(Zll_timeIter, curr_node, [i,j,k], ind)
                                force_interaction = 0

                            # In practice, since we made the hypothesis that VAC and VOIDS are sessile at 300K
                            # we never use the grad function and never access the else statement of the if... else
                            # loop above.
                        else :
                            # If the loop is sessile (100 loop)
                            force_interaction = 0

                        # ind is the place of the loop in the linked list ...
                        ind += 1

                        # The force due to the boundary is the force on the index curr_node.id
                        # of the list force_boundary
                        if (curr_node.val.size != -2):
                            force_boundary = force_boundary_l[curr_node.id]
                        else :
                            force_boundary = force_boundary_l_3d[curr_node.id]

                        # Total force exerced on the loop
                        force_tot = force_boundary + force_interaction

                        # Evolution of the position of the center of mass of the curr_node (loop)
                        # on the direction of the burger vector. Eq 9 PR 81, 224107
                        xi = uf.xsi()
                        # If it is a loop or an SIA defect
                        if not(-5 < curr_node.val.size <= -1.5559):
                            dzi = dt * (-curr_node.val.diffusivity(T) / (kB * T) * force_tot + \
                                            np.sqrt(2 * curr_node.val.diffusivity(T)/dt) * xi)
                        # If it is a
                        else :
                            D = curr_node.val.diffusivity(T)
                            # XSI = torch.tensor([uf.xsi(), uf.xsi(), uf.xsi()], dtype = dtype, device = device)
                            # dx = dt * (-1.0/(kB * T) * torch.tensordot(D, force_tot, 1) + \
                            #         torch.tensordot(D.mult(2.0/dt).sqrt(), XSI, 1))

                            dx = 0


                        # We add this evolution to the position marking the center of mass of the loop
                        # Z_new = Z + dzi*b/norm(b)
                        # The equation is not the same if the defect is a point vacancy.
                        #
                        b = torch.norm(curr_node.val.burger)
                        if not(-5 < curr_node.val.size <= -1.556):
                            curr_node.val.position = curr_node.val.position + \
                                    dzi.item()/b * curr_node.val.burger
                            # If the defect is a SIA, we change the degenerate direction
                            if (1.555 <= curr_node.val.size <= 1.556):
                                ind_sign = np.random.randint(0,4)
                                sign = np.array([[1,1,1], [1,-1,1], [1,-1,-1], [1,1,-1]])[ind_sign]
                                curr_node.val.dtensor, curr_node.val.burger = uf.degenerate_loop(curr_node.val.dtensor, \
                                                                                            curr_node.val.burger, sign)
                                curr_node.val.dtensor = uf.dipole_coord_analytical(curr_node.val.burger, \
                                                        curr_node.val.size, nu, mu, EOMs.a)
                        else :
                            curr_node.val.position = curr_node.val.position + dx



                        # Add the displacement in the list of
                        try :
                            DZI[curr_node.id] += torch.norm(dzi.item()/b * curr_node.val.burger).item()
                        except :
                            print('Node {} is not considered in the ATS process for t = {}s.'\
                                    .format(curr_node.id, ETA))
                        # Print control data

                        writing.control(curr_node, time_iter, [i,j,k], force_boundary, force_interaction, xi, t1, kB, T, 10)

                        # Save the forces during the process for post processing
                        F_BC[curr_node.id] = force_boundary
                        F_INT[curr_node.id] = force_interaction

                        # Let's move onto the next node
                        curr_node = copy.deepcopy(curr_node.next)

        # Is there a collision ?
        Zll = linked.collision(Zll)
        # Actualize the linked list System
        Zll = linked.actualize(Zll, Np, Nz, Nbr)

        # Add up the time to ETA: (when dt will be changing it will have an impact)
        ETA += dt

        # Formation volume associated with the configuration at time_iter
        for_volume = uf.formation_volume(Zll, nu, mu, EOMs.a)

        # Write down the state at time t in the out_... file
        Nloops_inter, Nloops_sessile = writing.write_pos_ll(Zll, time_iter + 1, example, Np, Nloops, T, FEA, dt, F_INT, F_BC, ETA, for_volume)

        if (example == 2):
            Ninit = Nmax
            Zll, Nmax = linked.insitu_TEM(Zll, dt, loops_data, Nmax)
            if Nmax != Ninit:
                dt = dt_init
	    

        if (Nloops_inter <= 1 or Nloops_inter == Nloops_sessile):
            break

        # Time the operation.
        print('TIME FOR 1 ITERATION : ', time.time() - t0_ti)
        print('\n')

    return Zll, DT


if __name__ == '__main__':
    t0 = time.time()
    Zll, DT = main(dt, FEA)

    # # Show the final state
    # for i in range(dim_Zllp):
    #     for j in range(dim_Zllp):
    #         for k in range(dim_Zllz):
    #             print(i,j,k)
    #             Zll[i,j,k].show()

    print('TOTAL TIME : ', time.time() - t0, 's')
