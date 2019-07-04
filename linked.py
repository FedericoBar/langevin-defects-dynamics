"""
Author Federico Baraglia
Date 2.5.2019

Functions useful for the use of linked lists objects

"""

import numpy as np
import EOM_struct as EOMs
import EOM_integrate as EOM
import copy
import torch
import useful_functions as uf
import time
import set_init


def gridalize(Z, Np, Nz, Nbr):
    """
    This function introduce the grid for GPU optimization
    It puts each defect in a cell of the gris
    Input : System at time t Z and size box N (Å), empty linked list system Zll
    Output : Linked system divided in cells
    """
    # Number of cell per sides
    dim_Zllp = Np//Nbr
    dim_Zllz = Nz//Nbr
    # Initialise the link list to an empty one.
    Zll = np.empty((dim_Zllp, dim_Zllp, dim_Zllz), dtype = EOMs.linklist)
    for i in range(dim_Zllp):
        for j in range(dim_Zllp):
            for k in range(dim_Zllz):
                Zll[i,j,k] = EOMs.linklist()

    # The number of loops is the length of Z...
    Nloops = len(Z)
    # For each node we find the cell it is meant to be on and link it to
    # the appropriate linked list
    for i_loop in range(Nloops):
        dislo_loop = EOMs.node(Z[i_loop], i_loop)
        i, j, k = associated_cell(dislo_loop, Nbr)

        Zll[i,j,k].link_node(dislo_loop)
    return Zll


def associated_cell(node, Nbr):
    """
    This function put the node in the right cell of the grid
    Input : node (loop position is the interesting input) and number of cells
            per side
    Output : [i,j,k] corresponding to the cell the node is in
    """

    i = int(node.val.position[0]//Nbr)
    j = int(node.val.position[1]//Nbr)
    k = int(node.val.position[2]//Nbr)
    res = [i,j,k]
    for x in range(3):
        # If the cell is close to a surface i=0 then int does not give
        # the correct answer
        if (node.val.position[x] < 0):
            res[x] -= 1
    return res[0], res[1], res[2]

def node_in_cell(node, cell_id, Nbr):
    """
    This function returns True if node is in the cell [i,j,k] False if not
    """
    i, j, k = associated_cell(node, Nbr)

    if (i == cell_id[0] and j == cell_id[1] and k == cell_id[2]):
        return True
    else:
        return False

def actualize(Zll, Np, Nz, Nbr):
    """
    This function changes the list for nodes that changed cells
    Input : linked list System
    Output : Refreshed linked list system
    """

    # We read all the linked lists, verifying for each node if they are in the right cell
    # Initialise the list containing the nodes not in their cells.
    out = []

    for i in range(Np//Nbr):
        for j in range(Np//Nbr):
            for k in range(Nz//Nbr):
                # Curr_list is the linked list associated with cell i j k
                curr_list = Zll[i,j,k]
                # curr_node will run through the nodes of curr_list
                curr_node = curr_list.head

                # New linked list = Zll[i,j,k] without the out nodes
                new_list = EOMs.linklist()

                # run through the nodes of curr_list
                while (curr_node != None):
                    # is the node stil in the adequate cell?
                    # If not add the node to the out nodes and not add it to the
                    # new list

                    if (not(node_in_cell(curr_node, [i,j,k], Nbr))):
                        out += [curr_node]
                    # If yes then add it to the new_list
                    else:
                        new_list.link_node(curr_node)

                    curr_node = copy.deepcopy(curr_node.next)

                # Change cell [i,j,k] with the new one
                Zll[i,j,k] = new_list

    # Replace the nodes in the right cell
    for node in out:
        i, j, k = associated_cell(node, Nbr)
        if (i < 0 or j < 0 or k < 0 \
        or i >= EOM.Np or j >= EOM.Np or k >= EOM.Nz):
            print('Node {} was removed because out of the BOX.'.format(node.id))
        # If it is not out, add it to the appropriate cell
        else :
            try :
                Zll[i,j,k].link_node(node)
            except :
                print('Node {} was removed because out of the BOX.'.format(node.id))

    return Zll

def collision(Zll):
    """
    This function joins every defect which is too close to another one
    This has to happen before actualise (we do not miss any collision)
    and we can use actualise to delete the collided defects.
    Input : System Zll, eps minimal distance between loops
    Output : New linked list system Zll
    """
    # Initialise the collided list... We will avoid removing twice the
    # same loop.
    collided = []


    for i in range(EOM.dim_Zllp):
        for j in range(EOM.dim_Zllp):
            for k in range(EOM.dim_Zllz):
                # Initialise the eaten list... it will avoid adding a non existing loop
                eaten = []

                # The closest loop is unlikely in cells that are more than 1
                # cell away
                NEIGHBOURS = neighbours([i,j,k], EOM.dim_Zllp, EOM.dim_Zllz, 1)

                # Common linked list method...
                curr_list1 = Zll[i,j,k]
                curr_node1 = curr_list1.head

                # New_list1 will contain the new linked list i j k without the annihilated loops
                new_list1 = EOMs.linklist()
                while (curr_node1 != None):
                    # Run the neighbours of cell [i,j,k]
                    for neigh in NEIGHBOURS:
                        # id of cell neigh
                        ii = neigh[0]
                        jj = neigh[1]
                        kk = neigh[2]
                        curr_list2 = Zll[ii,jj,kk]
                        curr_node2 = curr_list2.head
                        # New_list2 will contain the new linked list of cell ii jj kk without the annihilated loops
                        new_list2 = EOMs.linklist()
                        while (curr_node2 != None):
                            # too_close is the boolean representing collision
                            # smaller is the index of the smaller loop:
                            #           0 = curr_node1 .... 1 = curr_node2
                            too_close, smaller = curr_node1.collision(curr_node2)
                            if (too_close and curr_node1.id != curr_node2.id and [curr_node2.id, curr_node1.id] not in collided):
                                # Not to remove twice the same loop...
                                # It would work without but it is useless.
                                collided += [[curr_node1.id, curr_node2.id]]

                                # If we have 2 SIA's
                                if (curr_node1.val.size == 2 and curr_node2.val.size == 2):
                                    [curr_node1, curr_node2][1 - smaller].val.size = curr_node1.val.size + curr_node2.val.size
                                    [curr_node1, curr_node2][1 - smaller].val.dtensor = \
                                                            uf.dipole_coord_analytical([curr_node1, curr_node2][1 - smaller].val.burger, \
                                                            [curr_node1, curr_node2][1 - smaller].val.size, EOM.nu, EOM.mu, EOMs.a)

                                    deg_dir = np.random.choice([torch.tensor([1,1,1]), torch.tensor([1,1,-1])\
                                                            , torch.tensor([1,-1,1]), torch.tensor([-1,1,1])])

                                    [curr_node1, curr_node2][1 - smaller].val.dtensor, [curr_node1, curr_node2][1 - smaller].val.burger = \
                                                            uf.degenerate_loop([curr_node1, curr_node2][1 - smaller].val.dtensor, \
                                                            [curr_node1, curr_node2][1 - smaller].val.burger, deg_dir)

                                # Other cases
                                else :
                                    # Size of the new loop is the sum of the sizes
                                    [curr_node1, curr_node2][1 - smaller].val.size = curr_node1.val.size + curr_node2.val.size

                                    # Compute associated dipole tensor coordinates
                                    [curr_node1, curr_node2][1 - smaller].val.dtensor = \
                                                            uf.dipole_coord_analytical([curr_node1, curr_node2][1 - smaller].val.burger, \
                                                            [curr_node1, curr_node2][1 - smaller].val.size, EOM.nu, EOM.mu, EOMs.a)


                                # If we have a 100 SIA loop and another loop such that the bigger is the SIA and the result is a SIA
                                # defect
                                if (uf.collision_100SIA_to_SIA(curr_node1, curr_node2, [curr_node1, curr_node2][1 - smaller])):
                                    deg_dir = np.random.choice([torch.tensor([1,1,1]), torch.tensor([1,1,-1])\
                                                            , torch.tensor([1,-1,1]), torch.tensor([-1,1,1])])

                                    [curr_node1, curr_node2][1 - smaller].val.dtensor, [curr_node1, curr_node2][1 - smaller].val.burger = \
                                                            uf.degenerate_loop([curr_node1, curr_node2][1 - smaller].val.dtensor, \
                                                            [curr_node1, curr_node2][1 - smaller].val.burger, deg_dir)
                                # If the loops are very different in size the biggest eats the smallest
                                # If they are similar in size... the burger vector is the sum
                                # Keep that in mind

                                # Add the eaten loop to the eaten list.
                                eaten += [[curr_node1.id, curr_node2.id][smaller]]

                                # Print something to understand if there has been some collisions
                                print('Node ', [curr_node1, curr_node2][smaller].id, ' was annihilated by node ',\
                                                                        [curr_node1, curr_node2][1 - smaller].id)

                            if (not(curr_node2.id in eaten)):
                                new_list2.link_node(curr_node2)


                            # Go to next node on curr_list2
                            curr_node2 = curr_node2.next

                        # Refresh the ii jj kk linked list
                        Zll[ii,jj,kk] = new_list2

                    if (not(curr_node1.id in eaten)):
                        new_list1.link_node(curr_node1)

                    # Go to next node on curr_list1
                    curr_node1 = curr_node1.next

                # Refresh the i j k linked list
                Zll[i,j,k] = new_list1
    return Zll

def insitu_TEM(Zll, dt, loops_data, Nmax):
    """
    This function adds defects according to Daniel's DFT data.
    It has to be ran at every time step before collision and actualize
    Input : Linked list system Zll and actual time step dt and DFT data
    Output : New inked list system with the new defects
    """
    # How many ions produce cascade during 1 sec on the sample ?
    ion_per_sec = EOM.ion_flux * EOM.Np**2 # ions per sample surface per seconds
    # Idem but in one time step ?
    ion_per_dt = ion_per_sec * dt # ions per sample surface per dt
    # Since we should not be seeing a cascade every dt, we introduce
    # a probability for having a collision cascade taking place
    a = np.random.random()
    if a < ion_per_dt:
        ion_per_dt = 1

        # # For testing...
        # ion_per_dt = 1

        new_loops = set_init.main(EOM.data_ex3, EOM.Np, ion_per_dt, EOM.nu, EOM.mu, loops_data)

        for loop in new_loops:
            # The maximum number of loops increase of 1
            # This will help us with the id's of each loop
            # Create the node associated with the loop
            node = EOMs.node(loop)
            # Associate a new id
            node.id = Nmax
            Nmax += 1
            i,j,k = associated_cell(node, EOM.Nbr)

            Zll[i,j,k].link_node(node)

            Zll = actualize(collision(Zll), EOM.Np, EOM.Nz, EOM.Nbr)

    return Zll, Nmax


def cell_in_box(cell, Np, Nz):
    """
    This function returns True if the cell is in the box False if not.
    """
    # Coordinates of the cell
    i = cell[0]
    j = cell[1]
    k = cell[2]
    # Check if the cell is in the box [0,n0]*[0,n0]*[0,n0]
    if (i < 0 or j < 0 or k < 0):
        return False
    if (i >= Np or j >= Np or k >= Nz):
        return False
    # Return according to the result...
    return True

def neighbours(cell_id, dim_p, dim_z, R_cut):
    """
    This function returns the cells that are in the cut off radius R_cut from cell i,j,k
    (we take the infinite norm for simplicity, for now...)
    Input : cell_id, (Np//Nbr, Nz//Nbr) (nbr of cells per side) and cut off radius
    Output : set of neighbours [[i1,j1,k1], [i2,j2,k2], ...]
    """
    # Coordinates of the cell
    i = cell_id[0]
    j = cell_id[1]
    k = cell_id[2]

    # Initialise the output (the neighbours)
    neigh = []

    # List the cells that are close enough from cell_id.
    # We use infinite norm to measure distance.
    # a cell that is less than R_cut away from cell_id is a neighbour
    close = [x for x in range(-R_cut, R_cut + 1)]
    for ii in close:
        for jj in close:
            for kk in close:
                cell = [i+ii, j+jj, k+kk]
                # Add to the neighbours only if cell is in the box
                if cell_in_box(cell, dim_p, dim_z):
                    neigh += [[i + ii, j + jj, k + kk]]

    return neigh

"""
The following is to check the insitu_TEM, collision and actualize functions...
"""

# if __name__ == "__main__":
#     Nmax = 100
#     Zll, loops_data = EOM.init(Nmax, 0)
#
#     for i in range(EOM.dim_Zll):
#         for j in range(EOM.dim_Zll):
#             for k in range(EOM.dim_Zll):
#                 print(i,j,k)
#                 curr_list = Zll[i,j,k]
#                 if (curr_list.head != None):
#                     curr_node = curr_list.head
#                     while (curr_node != None):
#                         print('node identity : ', curr_node.id)
#                         print('node position : ', curr_node.val.position)
#                         print('node size : ', curr_node.val.size)
#
#                         curr_node = curr_node.next


    # change = 0
    # for i in range(EOM.dim_Zll):
    #     for j in range(EOM.dim_Zll):
    #         for k in range(EOM.dim_Zll):
    #
    #             curr_list = Zll[i,j,k]
    #             if (curr_list.head != None):
    #                 curr_node = curr_list.head
    #                 while (curr_node != None):
    #                     if (change <= 2):
    #                         curr_node.val.position = torch.tensor([10*(change+1),10,10], dtype = torch.double)
    #                         change += 1
    #
    #                     curr_node = curr_node.next


    # print('\nNext, add and collide the grid\n')
    #
    # Zll, Nmax = insitu_TEM(Zll, EOM.dt, loops_data, Nmax)
    # Zll = collision(Zll)
    # Zll = actualize(Zll, EOM.N, EOM.Nbr)
    #
    # for i in range(EOM.dim_Zll):
    #     for j in range(EOM.dim_Zll):
    #         for k in range(EOM.dim_Zll):
    #             print(i,j,k)
    #             curr_list = Zll[i,j,k]
    #             if (curr_list.head != None):
    #                 curr_node = curr_list.head
    #                 while (curr_node != None):
    #                     print('node identity : ', curr_node.id)
    #                     print('node position : ', curr_node.val.position)
    #                     print('node size : ', curr_node.val.size)
    #
    #                     curr_node = curr_node.next
    #
    # print(Nmax)


#
