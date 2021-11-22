import psi4

# psi4.set_memory('500 MB')

h = psi4.geometry("""
H 
""")

psi4.set_options({'reference': 'uhf'})
print(psi4.energy('scf/cc-pvdz'))