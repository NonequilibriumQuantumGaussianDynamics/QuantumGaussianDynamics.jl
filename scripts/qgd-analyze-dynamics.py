import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 

import cellconstructor as CC
import cellconstructor.Phonons

import sys, os
import ase, ase.io


CMAP = colors.LinearSegmentedColormap.from_list(name = "custom", colors = ["white", "blue", "red", "yellow"])



def get_bymode_dynamics(posfile, dyn, save_dynamics=True):
    """
    """
    # Load the data of the dynamics
    dynamics_data = np.loadtxt(posfile)

    forcefile = posfile.replace(".pos", ".ext") 
    data_forces = np.loadtxt(forcefile)

    w, pols = dyn.DiagonalizeSupercell()
    super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
    m = np.tile(super_struct.get_masses_array(), (3, 1)).T.ravel()

    # Exclude pure translations
    w = w[3:]
    pols = pols[:, 3:]
    n_modes = len(w)

    # Get the dynamical vector
    nat = super_struct.N_atoms
    n_t = dynamics_data.shape[0]
    motion = np.zeros((n_t, nat *3), dtype=np.double)
    momentum = np.zeros((n_t, nat *3), dtype=np.double)
    forces = np.zeros((n_t, nat *3), dtype=np.double)
    motion[:, :] = dynamics_data[:, 1: nat * 3 + 1].reshape((n_t, nat*3))
    print(data_forces.shape, forces.shape, n_t, nat, dynamics_data.shape)
    forces[:, :] = data_forces[:n_t, :nat * 3]
    motion -= super_struct.coords.ravel() * CC.Units.A_TO_BOHR

    momentum[:, :] = dynamics_data[:, nat * 3 + 1: nat * 6 + 1].reshape((n_t, nat*3))

    motion[:, :] *= np.tile(np.sqrt(m), (n_t, 1))
    momentum[:, :] /= np.tile(np.sqrt(m), (n_t, 1))
    forces[:, :] /= np.tile(np.sqrt(m), (n_t, 1))

    if save_dynamics:
        structs = []
        for i in range(n_t):
            s = super_struct.copy()
            s.coords = dynamics_data[i, 1:nat*3+1].reshape((nat, 3))
            structs.append(s.get_ase_atoms())

        ase.io.write("dynamics_{}.xyz".format(posfile), structs)


    nmodes = len(w)
    final_data = np.zeros((n_t, 3*len(w)+1), dtype=np.double)
    final_data[:, 0] = dynamics_data[:, 0]
    final_data[:, 1: nmodes+1] = np.dot(motion, pols)
    final_data[:, nmodes+1: 2*nmodes + 1] = np.dot(momentum, pols)
    final_data[:, 2*nmodes+1:] = np.dot(forces, pols)
    return final_data


def plot_stress(posfile, ax=None):
    """
    Get the stress data from the posfile
    Posfile must end with .pos, but the stress is read from
    Note that the one with extension .str is used
    """

    posdata = np.loadtxt(posfile)
    stressfile = posfile.replace(".pos", ".str")
    stress_data = np.loadtxt(stressfile)
    n1 = stress_data.shape[0]
    n2 = posdata.shape[0]
    n_max = np.min([n1, n2])

    if ax is None:
        plt.figure()
        ax = plt.gca()

    Voigt_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]    

    for i in range(6):
        stress = stress_data[:n_max, i] * 29421.01569663126 # Hartree/Bohr^3 to GPa
        ax.plot(posdata[:n_max, 0], stress, label = "%s" % Voigt_labels[i])

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Stress (GPa)")
    ax.legend()





def compute_energy(w, per_mode_data):
    """
    Compute the energy contribution of the system on a per-mode basis.

    Parameters
    ----------
    w : array_like
        The frequencies of the modes
    per_mode_data : array_like
        The data of the dynamics on a per-mode basis
        of shape (n_t, 2*len(w)+1), containing
        the per mode displacement and momentum (mass rescaled)
    """
    n_t, _ = per_mode_data.shape
    nmodes = len(w)
    energy = np.zeros((n_t, len(w)), dtype=np.double)
    for i in range(len(w)):
        energy[:, i] += 0.5 * w[i]**2 * np.abs(per_mode_data[:, i+1])**2
        energy[:, i] += 0.5 * np.abs(per_mode_data[:, i+nmodes+1])**2

    return energy

def create_colormap(w, energies, w_start=0, w_end=200/CC.Units.RY_TO_CM, n_w = 200):
    """
    Given the energy per mode per time, create
    a colormap of the energy per mode.

    Parameters
    ----------
    w : array_like
        The frequencies of the modes
    energies : array_like
        The energy per mode per time
        of shape (n_t, len(w))
    w_start : float
        The starting frequency of the colormap (same units as w)
    w_end : float
        The ending frequency of the colormap (same units as w)
    n_w : int
        The number of frequencies to consider

    Returns
    -------
    energy_map : array_like
        The energy map of shape (n_t, n_w)
    """

    delta_w = (w_end - w_start) / n_w
    delta_t = energies[1, 0] - energies[0, 0]

    n_t, _ = energies.shape
    w_array = np.linspace(w_start, w_end, n_w)


    energy_map = np.zeros((n_t, n_w), dtype=np.double)
    n_w_map = len(w)
    w_map = np.tile(w_array, (n_t, 1))

    print(n_w_map, n_t, n_w, w_map.shape, energies.shape)

    for i in range(n_w_map):
        energy_map[:, :] += np.exp(- (w_map - w[i])**2 / (2 * delta_w**2)) * np.tile(energies[:, i], (n_w, 1)).T / np.sqrt(2 * np.pi * delta_w**2)

    return energy_map


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot.py <dynamical_matrix> <nqirr> <posfile.pos> <mode-ids> ... ")
        sys.exit(1)

    dynfname = sys.argv[1]
    nqirr = int(sys.argv[2])
    fname = sys.argv[3]

    modes = [int(i) for i in sys.argv[4:]]

    source_dir = os.path.dirname(fname)

    lbl_files = os.path.basename(fname)
    lbl_files = lbl_files.replace(".pos", "")

    dyn = CC.Phonons.Phonons(os.path.join(source_dir, dynfname), nqirr)
    data = get_bymode_dynamics(fname, dyn)

    
    w, pols = dyn.DiagonalizeSupercell()
    w = w[3:]
    pols = pols[:, 3:]

    energies = compute_energy(w, data)
    last_w = w[-1] * 1.05
    energy_map = create_colormap(w, energies, w_end = last_w)

    plot_stress(fname)

    for i in modes:
        freq = w[i] * CC.Units.RY_TO_CM

        plt.figure()
        plt.title("Mode %d (%.2f cm-1)" % (i, freq))

        plt.xlabel("Time (fs)")
        plt.ylabel("$R\sqrt{M}$ [a.u.]")
        plt.plot(data[:, 0], data[:, i+1], color  = "k")

        # Add the force on the other axis
        ax2 = plt.twinx()
        ax2.set_ylabel("$F/\sqrt{M}$ [a.u.]")
        ax2.plot(data[:, 0], data[:, i + 2*len(w) + 1], color = "r")
        # Set the yaxis as red
        ax2.yaxis.label.set_color("red")
        ax2.tick_params(axis='y', colors='red')

        plt.tight_layout()

    # Plot the colormap
    plt.figure()
    plt.imshow(energy_map.T, origin="lower", aspect='auto', extent=(data[0, 0], data[-1, 0], w[0] * CC.Units.RY_TO_CM, last_w * CC.Units.RY_TO_CM), cmap=CMAP)
    plt.colorbar()
    plt.xlabel("Time (fs)")
    plt.ylabel("Frequency (cm-1)")
    plt.tight_layout()
    plt.savefig("{}_energy_map.png".format(lbl_files))

    plt.show()








