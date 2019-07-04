import torch
import useful_functions as uf
import numpy as np
from math import pi

dtype = torch.double
device = torch.device('cpu')

"""
Definition of all the class structures we need and their methods.
node : element of a linked list. Has a value and points to a next node (None)
linklist : has a head (which is a node)
loop_obj : describe a dislocation loop and what can be done with it.
"""
# We have to define this constant here...
a = 3.16 # Lattice constant in Å (for W)
kB = 8.617e-5 # Boltzmann's constant eV/K

Em_sia = 0.3 # Activation energy for the diffusivity for self interstitial point defects
Em_vac = 1.729 # Activation energy for the diffusivity for vacancy point defects

# reduced planck constant
h_bar = 6.582119e-16 # eV.s


class node:
    """
    object node which constitues the linked list object
    only method for initialisation and troubleshooting
    """
    def __init__(self, value = None, index = 0):
        """
        Classis init method for node data structure
        """
        self.val = value
        self.next = None
        self.id = index

    def show(self):
        """
        Shows the value of the node and the Next node it is linked
        to
        """
        print(self.val)
        print(self.next)


    def collision(self, other):
        """
        Returns True if self - other < eps False if not
        also returns the smaller loop (0 for self 1 for other)
        -1 if there is no collision
        """
        pos_self = self.val.position
        size_self = self.val.size
        pos_other = other.val.position
        size_other = other.val.size

        # Euclidian distance between the loops
        dist = torch.norm(pos_self - pos_other)
        # They collide if they are less than 2 spans away from
        # each other
        eps = (torch.abs(size_self) + torch.abs(size_other)).to(torch.double)

        if (dist < eps):
            if (size_self > size_other):
                return True, 1
            if (size_self < size_other):
                return True, 0
            else :
                a = np.random.choice([0,1])
                return True, a
        else:
            return False, -1


class linklist:
    """
    Object linked list representing the defects in a cell of the grid
    """
    def __init__(self):
        """
        Classis init method for link list data structure
        """
        self.head = None

    def link_node(self, node_to_add):
        """
        Add a node to a list in the head position
        """
        new_node = node()
        new_node.val = node_to_add.val
        new_node.id = node_to_add.id

        new_node.next = self.head
        self.head = new_node

    def delink_node(self, node):
        """
        Remove a specific node from a list
        if the node is not in the list then it does nothing
        """
        curr_node = self.head
        prev_node = self.head
        while (curr_node != node and curr_node != None):
            prev_node = curr_node
            curr_node = curr_node.next
        if (prev_node != None and curr_node != None):
            prev_node.next = curr_node.next

    def show(self):
        """
        Shows the position of each node of the list
        from head to tail
        """
        if (self.head != None):
            curr_node = self.head
            while (curr_node != None):
                print('node identity : ', curr_node.id)
                print('node position : ', curr_node.val.position)
                print('node burger : ', curr_node.val.burger)
                print('node size : ', curr_node.val.size)
                print('node dtensor : ', curr_node.val.dtensor)

                curr_node = curr_node.next

    def redu(self, i_device = -1):
        """
        Reduction for a defect cluster according to NF 58 (2018) 126002
        """
        curr_node = self.head
        if (i_device > -1):
            curr_devi = torch.device('cuda:' + str(i_device))
        else:
            curr_devi = 'cpu'

        # Initialise the numerator of the reduced position
        r_redu_up = torch.zeros((3,), dtype = dtype, device = curr_devi)
        # Initialise the denominator of the reduced position
        r_redu_bo = 0
        # Initialise the reduced dipole tensor
        dt_redu = torch.zeros((3,3), dtype = dtype, device = curr_devi)
        while (curr_node != None):
            dt_redu += curr_node.val.dtensor
            frob = torch.frobenius_norm(dt_redu)

            r_redu_up += frob * curr_node.val.position
            r_redu_bo += frob

            curr_node = curr_node.next

        # r_redu is the ratio between numerator and denominator
        r_redu = r_redu_up/r_redu_bo
        # Create the associated loop.
        loop_redu = loop_obj(r_redu, torch.tensor([0,0,0], dtype = dtype, device = curr_devi), dt_redu)
        return loop_redu

class loop_obj:
    """
    Object Loop (It can also be a defect)
    Functions (to come) : absorption...

    Sort of defects:
        - Loop (Position, burger vector, radius, dipole tensor)
        - Void (sessile)
        - Vacancy
        - SIA
    """
    def __init__ (self, position=torch.tensor([0,0,0], dtype = dtype, device = device),\
     burger = torch.tensor([0,0,0], dtype = dtype, device = device),\
     dtensor = torch.zeros((3,3), dtype = dtype, device = device),\
     radius = torch.tensor([1], dtype = dtype, device = device)):
    # Init method... if there are no input, the loop is initialised with 0 position, burger vector
    # and dipole tensor.
        self.position = position # Must be a 1*3 torch.tensor
        self.burger = burger # Must be a 1*3 torch.tensor
        self.dtensor = dtensor # Must be a 3*3 torch.tensor
        self.size = radius # Must be a 1*1 torch.tensor

    def diffusivity(self, T):
        # Computation of the diffusivity according to
        # New Journal of Physics,19, 073024 (2017)
        # If the size is more than 5 Angstroms then the defect is a loop (7atoms)
        theta_D = 400
        nu_0 = h_bar/(2*pi*kB)*theta_D*np.exp(-theta_D/T)
        D0_vac = 3/4*a**2*nu_0 # Diffusivity constant for VAC
        D0_sia = 3/4*a**2*nu_0 # Diffusivity constant for SIA

        # If loop
        # If the defect is more than 2 SIA then it is a SIA loop
        # In the VAC case we can have voids between VAC and VAC loops
        if (self.size > 1.5559 or self.size < -5):
            burgera = torch.abs(self.burger)
            # If 111 loop
            if (burgera.equal(torch.tensor([0.5, 0.5, 0.5], dtype = dtype, device = device))\
    or 0.5773 < burgera[0] < 0.5775):
                return 176*np.sqrt(85**2 + T**2)/np.sqrt(uf.num_atoms(self.size, self.burger, a))*1e8
            # Else 100 loop
            else:
                return 0
        # Not a loop
        else :
            # VAC
            if (self.size >= -1.556 and self.size <= -1.555):
                # return D0_vac*np.exp(-Em_vac/(kB*T))*torch.eye(3)
                # It would seem that this expression is also true in the SIA defect case
                return 0
            # SIA
            if (self.size <= 1.556 and self.size >= 1.555):
                # return D0_sia*np.exp(-Em_sia/(kB*T))
                # At 300K VAC are sessile
                return 176*np.sqrt(85**2 + T**2)/np.sqrt(uf.num_atoms(torch.abs(self.size), a))*1e8
            # VOID # At 300K VOIDs are sessile
            else :
                return 0

    def kind(self, T):
        # This method returns the nature of the loop
        # SIA_g - SIA_s - VAC_g - VAC_s
        if (self.size < 0):
            if (self.diffusivity(T) == 0):
                return 'VAC_s'
            else :
                return 'VAC_g'
        else :
            if (self.diffusivity(T) == 0):
                return 'SIA_s'
            else :
                return 'SIA_g'

        return nature

    def __add__(self, other):
        # Method to define +
        """
        /\/\/\
        obsolete with the linked list system...
        /\/\/\
        """
        result = loop_obj()
        # The position of the resulting loop is the position of one of the two loops
        # In our case we use the first.
        result.position = self.position
        # If the nature are different then this is no longer valid
        result.size = self.size + other.size
        # The resulting burger vector is the burger vector from the biggest loop
        result.burger = self.burger * (abs(other.size) <= abs(self.size)) + \
                        other.burger * (abs(other.size) > abs(self.size))

        # We compute the dipole tensor of the resulting loop
        result.dtensor = uf.dipole_coord_analytical(result.burger, result.size, nu, mu)

        # Return the resulting
        return result
#
# r = torch.tensor([1,1,1], dtype = dtype)
# s = torch.tensor([81], dtype = dtype)
# b = torch.tensor([0.5,0.5,0.5], dtype = dtype)
# P = torch.tensor([[28496,-9900,9900]\
#                 ,[-9900,28496,-9900]\
#                 ,[9900,-9900,28496]], dtype = dtype)
# defect = loop_obj(r,b,P,s)
# S = uf.define_S(0.278,1.0065)
# print(uf.formation_volume_elem(defect, S, 3.16))
