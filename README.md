# Year 5 MMath Dissertation on Mathematical Quantum Chemistry

Code to accompany my disseration on Mathematical Quantum Chemistry.

The `rhf.py` file contains my own implementation of Restricted Hartree-Fock with the STO-3G basis set. This is adapted from https://medium.com/analytics-vidhya/practical-introduction-to-hartree-fock-448fc64c107b, but with a number of improvements and differences to make sure it runs smoothly for atoms as well as simple molecules. This implementation is only completely valid for closed-shell systems up to subshell 2s in the basis. Lithium does not work as it is open-shell, and Beryllium requires the addition of 2p basis orbitals to work.

The `figures.ipynb` is a Jupyter Notebook with code for generating all figures and tables for the Comparision of Methods chapter in the written report. This includes:
- Tables of ground state energies for first order PT, variational PT, ROHF with STO-3G and cc-pVQZ basis, and experimental energies
- Plots to compare first order PT and ROHF (STO-3G) for different $Z$ values
- Spectral gap plots for Li-F to match up experimental energy level data and the PT model, which should be exact in the large $Z$ limit
- Wave function densities and their radial probability plots for ROHF and PT model methods, for the first four elements (H, He, Li, Be)

All Hartree-Fock calculations are performed using the Python API module of psi4, an ab initio computational chemistry package (https://psicode.org/).
Note if you wish to download the Python module psi4 on a Windows computer using conda, there are some extra annoyances (see: http://forum.psicode.org/t/how-to-run-psi4-on-windows-10/2174/20). Simple solution for downloading psi4 1.4 into Python 3.8 using conda is to use `conda install psi4 python=3.8 -c psi4 -c conda-forge` within the anaconda comman prompt, with the environoment of choice currently active.
