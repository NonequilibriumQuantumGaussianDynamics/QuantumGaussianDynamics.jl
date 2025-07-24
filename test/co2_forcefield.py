import numpy as np
import sys, os
import ase, ase.calculators
from julia.api import Julia
from ase.calculators.calculator import Calculator
import cellconstructor as CC, cellconstructor.Structure
import cellconstructor.calculators

import sscha, sscha.Ensemble, sscha.SchaMinimizer
import sscha.Relax

jl = Julia(compiled_modules=False)

import julia, julia.Main
julia.Main.include("co2_ff.jl")

class CO2(Calculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, *args, **kwargs):
        super().calculate(atoms, *args, **kwargs)

        coords = atoms.get_positions().ravel()
        energy, forces = julia.Main.py_force_field(coords)

        self.results = {
                "energy": energy,
                "forces": forces
                }

def relax():
    struct = CC.Structure.Structure(3)
    struct.coords[0, 0] = -1.119
    struct.coords[2, 0] = 1.119
    struct.atoms = ["O", "C", "O"]
    struct.unit_cell = np.eye(3) * 10.0
    struct.has_unit_cell = True
    struct.build_masses()

    calc = CO2()

    relax = CC.calculators.Relax(struct, calc)
    relaxed_struct = relax.static_relax()
    relaxed_struct.save_scf("co2.scf")

    harm_dyn = CC.Phonons.compute_phonons_finite_displacements(relaxed_struct, 
                                                               calc, 
                                                               supercell=(1,1,1))
    harm_dyn.ForcePositiveDefinite()
    harm_dyn.Symmetrize()
    harm_dyn.save_qe("co2_dyn")

    # Perform a SSCHA
    ensemble = sscha.Ensemble.Ensemble(harm_dyn, 0)
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
    relax = sscha.Relax.SSCHA(minim, calc, N_configs=600, max_pop=8, save_ensemble=False)
    relax.relax()
    relax.minim.dyn.save_qe("co2_sscha")


if __name__ == "__main__":
    relax()

