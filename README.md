# ParMaCH - Parameterized model of a cooling magma chamber
1D model modelling crystal nucleation, growth, settling in an injected body of magma

Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 4) 
- Author: Vít Beran
- Affiliation: Department of Mathematics and Physics, Charles University, Prague
- Run with: python3 main.py (--flags of your choice)
- Run 'python3 main.py --help' for more information.

Special requirements: 
- Python version: Python 3.10. and higher
- numba (njit, jitclass), h5py, objgraph (memory precaution), tqdm

Contents:
- main.py:   main unit of the model, model setup, initilization
- mFunc.py:  auxiliary functions, custom crystal population classes, population dynamics solvers
- mPhase.py: individual phases of the computational process
- m1DHE.py:  1D heat equation module, grid setup & alteration, implicit coupling
- mGL.py:    implementation of the Grossmann-Lohse theory, iterative regime detector
- mPar.py:   constants, material parameters, initialization of jit-classes
- mJW94.py:  auxiliary implementation of the dimensionless simplified model by Jarvis & Woods (1994)
- mPost.py:  post-processing module, full visualization of the computed model
- mPlot.py:  visualization of CSDs, 1D heat equation solution

Deprecated:
- mMisch.py: miscellaneous functions, experimenting with different settling models

Parametric search: concurrent runs & visualization:
- paper_landscape.sh
- mGrid.py

Supported features:
- Vigorous and turbulent thermal Rayleigh-Benárd convection (Grossman-Lohse 2000)
- Crystal settling in a convecting fluid (Martin and Nokes 1989, Patočka et al. 2022)
- Crystal size distribution function approach (Randolph and Larsen 1971, Marsh 1988)
- Experimentally supportd kinetic law of crystal growth and nucleation (Hort 1997, Couch et al. 2003)
- 0D energy balance | 1D coupling with host rock (effective conductivity approach)
- Binary eutectic system Anorthite-Diopside (Courtial et al. 2000, Gale et al. 2008, Giordano et al. 2008, Krattli and Schmidt 2021)

Remark: as of 6/30/2026, the model runs in the quasi-stationary limit (one timescale approach not implemented yet).

Examples with explanations:
0D reference run A: python main.py --SOLVER=0 --input_hflux --hflux=200 --hfluxSI --V0HG97=1.e-6 --N0HG97=1.e3 --XL0=0.75 --ratio=10000 --HE_CSTM --SED_METHOD=3 --nu --Ti97=0.92 --Tg97=0.95

> SOLVER=0 (0D energy balance), input_hflux (model supplied with initial heat flux), hflux=200 (initial heat flux = 200), hfluxSI (W/m2), V0HG97 (growth rate amplitude),
  N0HG97 (nucleation rate amplitude), XL0 (initial compositon %wt An), ratio=10000 (estimate for the initial time step), HE_CSTM (heat flux follows an empirically prescribed decay),
  SED_METHOD=3 (invoke the crystal tracking algorithm for the population dynamics), nu (thermally-dependent viscosity), Ti97 (nucleation rate peak, nucleation lag), Tg97 (growth rate peak)

0D reference run B: python main.py --SOLVER=0 --input_hflux --hflux=200 --hfluxSI --V0HG97=1.e-6 --N0HG97=1.e3 --XL0=0.75 --ratio=1000 --HE_CONST --SED_METHOD=3 --nu --Ti97=0.81 --Tg97=0.93 

> SOLVER=0 (0D energy balance), input_hflux (model supplied with initial heat flux), hflux=40 (initial heat flux = 40), hfluxSI (W/m2), V0HG97 (growth rate amplitude),
  N0HG97 (nucleation rate amplitude), XL0 (initial compositon %wt An), ratio (estimate for the initial time step), HE_CONST (constant heat flux), SED_METHOD=3 (invoke the crystal tracking       algorithm for the population dynamics), nu (thermally-dependent viscosity), Ti97 (nucleation rate peak, nucleation lag), Tg97 (growth rate peak)

1D reference simulation: python main.py --SOLVER=1 --V0HG97=1.e-8 --XL0=0.75 --SED_METHOD=3 --nu --H0=1000 --suc=15.0

> SOLVER=1 (1D heat equation), V0HG97 (growth rate amplitude), XL0 (initial composition %wt An), SED_METHOD (crystal tracking algorithm), nu (thermally-dependent viscosity),
  H0 (initial height of the chamber), suc (initial degree of superheating)
