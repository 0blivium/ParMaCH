# ParMaCH - Parameterized model of a cooling magma chamber
1D model modelling crystal nucleation, growth, settling in an injected body of magma

Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 4) 
- Author: MSci. Vít Beran
- Run with: python3 main.py (--flags of your choice)
- Run 'python3 main.py --help' for more information.

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

Remark: as of 6/30/2026, the model runs in the quasi-stationary limit.
