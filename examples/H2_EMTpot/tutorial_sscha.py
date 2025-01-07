import ase
from ase import Atoms
from ase.build import molecule
import numpy as np
import sys
import os
# Import the cellconstructor library to manage phonons
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Structure, cellconstructor.calculators
from matplotlib import pyplot as pl
import copy
from ase.calculators.emt import EMT
from ase.optimize import BFGS

structure = CC.Structure.Structure()
structure.read_generic_file('scf.in')
n1 = 1
supercell = (n1,1,1)

calc = EMT()

crystal = Atoms(structure.atoms, positions=structure.coords, pbc=True, cell = structure.unit_cell)
crystal.calc = calc

opt    =BFGS(crystal)
opt.run(fmax=1e-4)
print(opt.atoms.positions)

structure.coords =  opt.atoms.positions

dyn = CC.Phonons.compute_phonons_finite_displacements(structure, calc, supercell = supercell)
dyn.Symmetrize()
dyn.save_qe("dyn")
dyn.ForcePositiveDefinite()


