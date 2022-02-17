""" 
Restricted Hartree-Fock with STO-nG basis set
Adapted from Laksh (2019) with page references to Szabo & Ostlund 

"""

# Imports
from importlib.util import module_for_loader
import numpy as np
from scipy.special import erf

### Integrals between Gaussian orbitals (S&O pp410-416) ###

def gauss_product(gauss_A, gauss_B):
    """
    Calculate the product of two GTOs A and B 
    Input: gauss_A, gauss_B are both tuples containing the exponent and centre for GTO A and B
    Output:
    p: sum exponents
    diff: distance between centres Ra and Rb
    K: proportionality constant and normalization
    Rp: new centre for Gaussian product
    """
    a, Ra = gauss_A
    b, Rb = gauss_B
    p = a + b
    diff = np.linalg.norm(Ra-Rb)**2
    N = ((4*a*b)/np.pi**2)**0.75     # normalization absorbed into K
    K = N*np.exp(-a*b/p * diff)
    Rp = (a*Ra + b*Rb)/p
    
    return p, diff, K, Rp

# Overlap integral
def overlap(A, B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (np.pi/p)**1.5
    return prefactor*K

# Kinetic integral
def kinetic(A, B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (np.pi/p)**1.5
    a, Ra = A
    b, Rb = B
    reduced_exponent = a*b/p
    return reduced_exponent*(3-2*reduced_exponent*diff)*prefactor*K

# F0 Boys function to calculate potenetial and e-e repulsion integrals
def F0(z):
    if z == 0:
        return 1
    else:
        return (0.5*(np.pi/z)**0.5)*erf(z**0.5)
    
# Nuclear attraction integral
def potential(A, B, molecule, atom_idx):
    charge_dict = {'H':1, 'He':2, 'Li':3, 'Be':4}
    p, diff, K, Rp = gauss_product(A, B)
    Rc = molecule['coordinates'][atom_idx] # position of atom C
    Zc = charge_dict[molecule['atoms'][atom_idx]] # charge of atom C
    
    return -2*np.pi*Zc*K/p * F0(p*np.linalg.norm(Rp-Rc)**2)

# (AB|CD) two-electron integral
def multi(A,B,C,D):
    p, diff_ab, K_ab, Rp = gauss_product(A,B) 
    q, diff_cd, K_cd, Rq = gauss_product(C,D)
    # note p=a+b, q=c+d
    multi_prefactor = 2*np.pi**2.5 / ((p*q*(p+q)**0.5))
    return multi_prefactor*K_ab*K_cd*F0(p*q/(p+q)*np.linalg.norm(Rp-Rq)**2)


def integral_calc(molecule, alpha, D):
    """
    Computate all pre-main SCF loop integrals

    Input:
    molcule: dictionary containing information about the molecule including 'N_atoms', 
            'atoms' (list of atom symbols), and 'coordinates' (array of atom centres)
    alpha: array of contraction exponents for STO-nG basis
    D: array of contraction coefficients for STO-nG basis
    
    Output:
    B: basis size
    S: BxB array containing overlap integrals
    H_core: BxB array containing H_core integrals
    multi_elec_tensor: BxBxBxB array containing two-electron integrals (ab|cd)
    """
    
    if molecule['N_atoms'] == 1:
        # Atom zeta values from Hehre et al., 1970
        zeta_dict = {'H':[1.24], 'He':[1.69], 'Li':[2.69,0.80], 'Be':[3.68,1.15]}
    else:
        # Molecule zeta values from Hehre et al., 1969
        zeta_dict = {'H':[1.24], 'He':[2.0925], 'Li':[2.69,0.80], 'Be':[3.68,1.15]}
    max_quantum_number = {'H':1, 'He':1, 'Li':2, 'Be':2}

    # Basis set size
    B = 0
    for atom in molecule['atoms']:
        B += max_quantum_number[atom]

    # Number of gaussians used to form a contracted gaussian orbital
    STOnG = D.shape[1]

    # Initialize matrices
    S = np.zeros((B,B))
    T = np.zeros((B,B))
    V = np.zeros((B,B))
    multi_elec_tensor = np.zeros((B,B,B,B))

    # Iterate over atoms
    for idx_a, val_a in enumerate(molecule['atoms']):
        Ra = molecule['coordinates'][idx_a] # get centre
        
        # Iterate over quantum number orbitals (1s, 2s, etc)
        for m in range(max_quantum_number[val_a]):
            # Get contraction coeffs (d), zeta, and then scale exponents (p158)
            d_vec_m = D[m]
            zeta = zeta_dict[val_a][m]
            alpha_vec_m = alpha[m]*zeta**2
            
            # Iterate over contraction coeffs
            for p in range(STOnG):
                
                # Iterate over atoms again
                for idx_b, val_b in enumerate(molecule['atoms']):
                    Rb = molecule['coordinates'][idx_b]
                    for n in range(max_quantum_number[val_b]):
                        d_vec_n = D[n]
                        zeta = zeta_dict[val_b][n]
                        alpha_vec_n = alpha[n]*zeta**2
                        
                        for q in range(STOnG):
                            # Correct indexing for Python
                            a = (idx_a+1)*(m+1)-1
                            b = (idx_b+1)*(n+1)-1
                            
                            # Generate overlap, kinetic and potential matrices
                            S[a,b] += d_vec_m[p]*d_vec_n[q]*overlap((alpha_vec_m[p],Ra), (alpha_vec_n[q],Rb))
                            T[a,b] += d_vec_m[p]*d_vec_n[q]*kinetic((alpha_vec_m[p],Ra), (alpha_vec_n[q],Rb))
                            
                            for i in range(molecule['N_atoms']):
                                V[a,b] += d_vec_m[p]*d_vec_n[q]*potential((alpha_vec_m[p],Ra), (alpha_vec_n[q],Rb), molecule, i)
                                
                            # 2 more iterations to get the multi-electron-tensor
                            for idx_c, val_c in enumerate(molecule['atoms']):
                                Rc = molecule['coordinates'][idx_c]
                                for k in range(max_quantum_number[val_c]):
                                    d_vec_k = D[k]
                                    zeta = zeta_dict[val_c][k]
                                    alpha_vec_k = alpha[k]*zeta**2
                                    for r in range(STOnG):
                                        for idx_d, val_d in enumerate(molecule['atoms']):
                                            Rd = molecule['coordinates'][idx_d]
                                            for l in range(max_quantum_number[val_d]):
                                                d_vec_l = D[l]
                                                zeta = zeta_dict[val_d][l]
                                                alpha_vec_l = alpha[l]*zeta**2
                                                for s in range(STOnG):
                                                    c = (idx_c+1)*(k+1)-1
                                                    d = (idx_d+1)*(l+1)-1
                                                    multi_elec_tensor[a,b,c,d] += d_vec_m[p]*d_vec_n[q]*d_vec_k[r]*d_vec_l[s]*(
                                                    multi((alpha_vec_m[p],Ra), (alpha_vec_n[q],Rb),
                                                        (alpha_vec_k[r],Rc), (alpha_vec_l[s],Rd)))                                                   
    # Form H_core
    H_core = T + V

    return B, S, H_core, multi_elec_tensor

# Symmetric orthogonalization of basis (S&O p143)
def sym_orth(S):
    eval_S, U = np.linalg.eig(S)
    diag_S = U.T@S@U
    diag_S_minushalf = np.diag(np.diagonal(diag_S)**-0.5)
    X = U@diag_S_minushalf@U.T
    return X

# Calculation of matrix G
def G_calc(B, P, multi_elec_tensor):
    G = np.zeros((B,B))
    for i in range(B):
        for j in range(B):
            for k in range(B):
                for l in range(B):
                    G[i,j] += P[k,l]*(multi_elec_tensor[i,j,k,l]-0.5*multi_elec_tensor[i,k,l,j])
    return G

# Electronic energy (S&O p176)
def electronic_energy(P, H_core, F):
    return np.trace(1/2 * (P@(H_core+F)))

# Nuclear repulsion energy (returns zero if atom not molecule)
def nuclear_energy(molecule):
    energy = 0
    charge_dict = {'H':1, 'He':2, 'Li':3, 'Be':4, 'C':5, 'N':6, 'O':8, 'F':9, 'Ne':10}
    if molecule['N_atoms'] > 1:
        for idx_a in range(molecule['N_atoms']):
            atom_a = molecule['atoms'][idx_a]
            coord_a = molecule['coordinates'][idx_a]
            for idx_b in range(idx_a+1, molecule['N_atoms']):
                atom_b = molecule['atoms'][idx_b]
                coord_b = molecule['coordinates'][idx_b]
                R_ab = np.linalg.norm(coord_a-coord_b)
                energy += charge_dict[atom_a]*charge_dict[atom_b] / R_ab
    return energy

############ SCF Algorithm ############
def scf(molecule, alpha, D, threshold=10**-6, max_it=250):

    B, S, H_core, multi_elec_tensor = integral_calc(molecule, alpha, D) # Integral calculations 
    STOnG = D.shape[1]
    X = sym_orth(S) # orthogonalization

    # Initial guess at density matrix P
    P = np.zeros((B,B))
    P_old = np.zeros((B,B))

    ## Main SCF loop iterative process ##
    P_diff = 100
    it = 0
    while P_diff > threshold:
        # Calculate Fock matrix with guess P
        G = G_calc(B, P, multi_elec_tensor)
        F = H_core + G # Fock matrix

        # Calculate Fock matrix in orthogonalized basis
        F_prime = X.T@F@X
        eval_F_prime, C_prime = np.linalg.eig(F_prime)

        # Ensure correct ordering of eigenvals and eigenvecs
        idx = eval_F_prime.argsort()
        eval_F_prime = eval_F_prime[idx]
        C_prime = C_prime[idx]
        
        C = X@C_prime # orbital coeffs for basis set

        # Form new density matrix P
        for i in range(B):
            for j in range(B):
                for a in range(int(molecule['N_elecs']/2)): # only sum over electron pairs (for RHF)
                    P[i,j] = 2*C[i,a]*C[j,a]
        
        P_diff = np.sqrt(np.sum((P_old-P)**2)/B**2) # Difference between old and new density matrices
        P_old = P.copy()

        it += 1
        if it > max_it:
            print('Maximum number of iterations reached \n')
            print('Current number of iterations is {}'.format(it))
            print('Current approximation for orbitals energies are {} Hartrees'.format(eval_F_prime))
            print(f'Current orbital matrix is: \n\n{C}')
            print('Current total electronic energy: {} \n'.format(elec_energy(P, H_core, F)))
            return 


    print('STO{}G Restricted Closed Shell HF algorithm for {} took {} iterations to converge \n'.format(STOnG, molecule['name'], it))
    print(f'The orbital matrix is: \n\n C = {C} \n')

    elec_energy = electronic_energy(P, H_core, F)
    nuc_energy = nuclear_energy(molecule)
    energy = elec_energy + nuc_energy
    if molecule['N_atoms'] > 1:
        print('Electronic energy: {} \n'.format(elec_energy))
        print('Nuclear repulsion energy: {} \n'.format(nuc_energy))
    print('Total energy: {} \n'.format(energy))

### Basis set variables for STO3G ###
# Coefficient values from Hehre et al., 1969

# Gaussian orbital exponents
alpha = np.array([[0.109818, 0.405771, 2.22766],  #1s
                [0.0751386, 0.231031, 0.994203]]) #2sp

# Gaussian contraction coeffs
D = np.array([[0.444635, 0.535328, 0.154329],   #1s
            [0.700115, 0.399513, -0.0999672]])  #2s


### Molecule & atom information ###
HeHplus = {'name': 'HeH+','N_elecs': 2, 'N_atoms':2, 'atoms': ['He', 'H'], 'coordinates': np.array([[0,0,0], [0,0,1.4632]], dtype=float)}
H2 = {'name': 'H2','N_elecs': 2, 'N_atoms':2, 'atoms': ['H', 'H'], 'coordinates': np.array([[0,0,0], [0,0,1.4]], dtype=float)}

H = {'name': 'H', 'N_elecs': 1, 'N_atoms':1, 'atoms': ['H'], 'coordinates': np.array([[0,0,0]], dtype=float)}
He = {'name': 'He', 'N_elecs': 2, 'N_atoms':1, 'atoms': ['He'], 'coordinates': np.array([[0,0,0]], dtype=float)}
Li = {'name': 'Li', 'N_elecs': 3, 'N_atoms':1, 'atoms': ['Li'], 'coordinates': np.array([[0,0,0]], dtype=float)}
Be = {'name': 'Be', 'N_elecs': 4, 'N_atoms':1, 'atoms': ['Be'], 'coordinates': np.array([[0,0,0]], dtype=float)}


### Perform SCF algorithm ###
scf(molecule=HeHplus, alpha=alpha, D=D)
