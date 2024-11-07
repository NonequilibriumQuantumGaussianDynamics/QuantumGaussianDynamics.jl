import ase, ase.calculators
import numpy as np
from ase.calculators.calculator import Calculator


class Harmonic3D(Calculator):
    def __init__(self, k1, k2, k3, *args, **kwargs):
        print("Harmonic3D.__init__: kwargs = ", kwargs)
        super().__init__(*args, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, *args,  **kwargs):
        super().calculate(atoms, *args, **kwargs)
        
        # Compute the harmonic energy
        coords = atoms.get_positions()
        energy = 0.5 * self.k1 * np.sum(coords[:, 0]**2) + \
                 0.5 * self.k2 * np.sum(coords[:, 1]**2) + \
                 0.5 * self.k3 * np.sum(coords[:, 2]**2)

        # Compute the harmonic forces   
        forces = np.zeros_like(coords)
        forces[:, 0] = -self.k1 * coords[:, 0]
        forces[:, 1] = -self.k2 * coords[:, 1]
        forces[:, 2] = -self.k3 * coords[:, 2]

        stress = np.zeros(6, dtype = np.double)

        self.results = {"energy": energy,
                        "forces": forces,
                        "stress": stress}
