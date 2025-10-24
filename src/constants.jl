"""
Basic constants necessary for conversions are defined
"""

const SMALL_VALUE::Float64 = 1e-8
const THR_ACOUSTIC::Float64 = 1e-1
const CONV_FS::Float64 = 0.048377687 # to femtoseconds
const CONV_RY::Float64 = 0.0734985857 # eV to Ry
const CONV_BOHR::Float64 = 1.8897261246257702 # Angstrom to Bohr
const CONV_MASS::Float64 = 911.444175 # amu to kg
const CONV_EFIELD::Float64 = 2.7502067*1e-7 #kVcm to E_Ry
const CONV_FREQ::Float64 = 4.83776857*1e-5 #THz to w_Ry

export SMALL_VALUE
export CONV_FS
export CONV_RY
export CONV_BOHR
export CONV_EFIELD
