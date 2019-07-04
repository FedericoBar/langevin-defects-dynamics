import torch.cuda
import numpy as np
import EOM_struct as EOMs
import copy

def broadcast_system(system, nbr_devices):
    """
    This function broadcast the system onto the GPU's
    """
    # Number of cell per sides
    dim_system = system.shape[0]

    # Initialise the replicas as an empty numpy array
    # dimension = nbr_devices used
    rep_system = np.empty((nbr_devices, ), dtype = list)
    for i_device in range(nbr_devices):
        # Initialise the shape of the replicas
        rep_system[i_device] = np.empty_like(system)

    # Initialisation saves us some computing time

    # For each cell, we create the same cell on all the replicas (of Zll on each GPU)
    for i in range(dim_system):
        for j in range(dim_system):
            for k in range(dim_system):
                # curr_list is the link list corresponding to the i,j,k cell
                curr_list = system[i,j,k]
                # curr_node will run through the node of curr_list
                # (the list in cell i,j,k)
                curr_node = curr_list.head

                # new_lists will contain all the replicas of curr_list
                # on the different GPU
                new_lists = [EOMs.linklist() for i in range(nbr_devices)]

                # Run through the nodes of the linked list curr_list
                while (curr_node != None):

                    # Fetch the components of the node for broadcasting
                    inter_node_position = curr_node.val.position
                    inter_node_dtensor = curr_node.val.dtensor
                    inter_node_burger = curr_node.val.burger

                    # Create the broadcasted tensor of each attributes
                    # It is a list of tensor equal to the input on each GPU
                    # we allocate on the GPU 1:nbr_devices
                    rep_pos = torch.cuda.comm.broadcast(inter_node_position, [i for i in range(nbr_devices)])
                    rep_dtensor = torch.cuda.comm.broadcast(inter_node_dtensor, [i for i in range(nbr_devices)])
                    rep_burger = torch.cuda.comm.broadcast(inter_node_burger, [i for i in range(nbr_devices)])

                    # Create the list containing the loop objects on each GPU
                    loops = [EOMs.loop_obj(rep_pos[i], rep_burger[i], rep_dtensor[i]) for i in range(nbr_devices)]
                    # Creating the list of nodes on each GPU
                    nodes = [EOMs.node(loops[i]) for i in range(nbr_devices)]


                    for i_device in range(nbr_devices):
                        # Append the new node to the new list for each GPU
                        new_lists[i_device].link_node(nodes[i_device])

                    # Go to next node
                    curr_node = copy.deepcopy(curr_node.next)

                # Create the replicated system
                for i_device in range(nbr_devices):
                    # For each device, the cell i,j,k wil be the one on the original system
                    rep_system[i_device][i,j,k] = new_lists[i_device]

    # rep_system is a list of system exactly as system but with data on other GPU
    return rep_system



#
