import ase, ase.calculators
import numpy as np
from ase.calculators.calculator import Calculator
import julia, julia.Main

# Load the force field
julia.Main.include("test_raman_perturbation.jl")

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

class Harmonic3D_ASR(Calculator):
    def __init__(self, k1, k2, k3, *args, **kwargs):
        print("Harmonic3D.__init__: kwargs = ", kwargs)
        super().__init__(*args, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.delta0 = np.array([1.0, 0.2, 0.0])

        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, *args,  **kwargs):
        super().calculate(atoms, *args, **kwargs)
        
        # Compute the harmonic energy
        coords = atoms.get_positions()
        delta = coords[0, :] - coords[1, :] - self.delta0


        energy = 0.5 * self.k1 * delta[0]**2 + \
                 0.5 * self.k2 * delta[1]**2 + \
                 0.5 * self.k3 * delta[2]**2

        # Compute the harmonic forces   
        forces = np.zeros_like(coords)
        forces[0, 0] = -self.k1 * delta[0]
        forces[0, 1] = -self.k2 * delta[1]
        forces[0, 2] = -self.k3 * delta[2]

        forces[1, :] = -forces[0, :]

        stress = np.zeros(6, dtype = np.double)

        self.results = {"energy": energy,
                        "forces": forces,
                        "stress": stress}

class Anharmonic3D_ASR(Calculator):
    def __init__(self, k1, k2, k3, delta1=1.0, delta2=0.2, delta3=0.0, anharmonicity=0.1, *args, **kwargs):
        print("Harmonic3D.__init__: kwargs = ", kwargs)
        super().__init__(*args, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.anharmonicity = anharmonicity
        self.delta0 = np.array([delta1, delta2, delta3])

        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, *args,  **kwargs):
        super().calculate(atoms, *args, **kwargs)
        
        # Compute the harmonic energy
        coords = atoms.get_positions()
        delta = coords[1, :] - coords[0, :] - self.delta0

        delta_norm = np.linalg.norm(delta)

        energy = 0.5 * self.k1 * delta[0]**2 + \
                 0.5 * self.k2 * delta[1]**2 + \
                 0.5 * self.k3 * delta[2]**2

        # Add the anharmonic energy
        energy += 0.25 * self.anharmonicity * delta_norm**4

        # Compute the harmonic forces   
        forces = np.zeros_like(coords)
        forces[0, 0] = (self.k1 + self.anharmonicity * delta_norm**2) * delta[0] 
        forces[0, 1] = (self.k2 + self.anharmonicity * delta_norm**2) * delta[1]
        forces[0, 2] = (self.k3 + self.anharmonicity * delta_norm**2) * delta[2]

        forces[1, :] = -forces[0, :]

        stress = np.zeros(6, dtype = np.double)

        self.results = {"energy": energy,
                        "forces": forces,
                        "stress": stress}


