# Example: H2

This example simulates a H2 molecule using the EMT potential as implemented in ASE. 
This potential is not accurate for H2, so it should not be used for serious calculation.

This example has multiple different calculation types:

## Raman via Impulsive Stimulated Raman Scattering

Raman spectrum can be computed within QuantumGaussianDynamics.jl in a moltitude of ways.
One of the easiest is to simulate the result of an Impulsive Stimulated Raman Scattering (ISRS) settings.
This experimental technique involves two laser pulses. 
An ultrashort (usually few fs long) *pump* pulse that exerts a force on the nuclei, followed by a *probe* pulse, that probes time-modulations of
the polarizzability of the system. The Fourier transform of the time-modulations offer the ISRS spectra. 
In the hypothesis of a pump pulse shaped as a Dirac delta, and assuming a linear response of the system on the pump intensity, we get a response that coincides with the one of an ideal Stimulated Raman.

For the ISRS (and any Raman based perturbation) to work, we need to compute the Raman tensor of the system, which is the derivative of the total polarizability as a function of the nuclear positions.
To simplify, we only account for the component of the Raman tensor along the stretching mode.

The setup file for this calculation is ``raman_isrs.jl``


