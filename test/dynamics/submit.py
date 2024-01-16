import cellconstructor as CC, cellconstructor.Phonons
import sscha, sscha.Ensemble, sscha.SchaMinimizer
import sscha.Utilities, sscha.Relax
import cellconstructor.calculators
import sys
import numpy as np
import copy
import glob
from ase.calculators.emt import EMT


TEMPERATURE = 0
N_CONFIGS = 1000
MAX_ITERATIONS = 50


dyn = CC.Phonons.Phonons('dyn',1)
dyn.ForcePositiveDefinite()
qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
qe_sym.SetupFromSPGLIB()
N_symm = qe_sym.QE_nsym

ensemble = sscha.Ensemble.Ensemble(dyn, TEMPERATURE)

# Define the minimization variables
minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)

# Save minimization data
iodata = sscha.Utilities.IOInfo()
iodata.SetupSaving('minim_data')

calculator = EMT()

# Relax across different ensemble
relax = sscha.Relax.SSCHA(minim, calculator, N_configs = N_CONFIGS,
                          max_pop = MAX_ITERATIONS,  save_ensemble = True)
relax.data_dir = 'data'
relax.setup_custom_functions(custom_function_post = iodata.CFP_SaveAll)

relax.relax(get_stress = True, ensemble_loc = 'sscha_ensemble')
relax.minim.finalize()
relax.minim.dyn.save_qe('final_result')





