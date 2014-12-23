Controlpy
=========

A library for commonly used controls algorithms (e.g. creating LQR controllers). An alternative to Richard Murray's "control" package -- however, here we do not require Slycot.

Current capabilities:

1) System analysis:
  a) Test whether a system is stable, controllable, stabilisable, observable, or stabilisable.
  b) Get the uncontrollable/unobservable modes
  c) Compute a system's controllability Gramian (finite horizon, and infinite horizon)
  d) Compute a system's H2 and Hinfinity norm

2) Synthesis
  a) Create continuous and discrete time LQR controllers
  b) Full-information H2 optimal controller
  c) H2 optimal observer
  d) Full-information Hinf controller


How to install
--------------
Install using pypi, or direct from the Github repository:

1) Clone this repository somewhere convenient: `git clone https://github.com/markwmuller/controlpy.git`
2) Install the package (we'll do a "develop" install, so any changes are immediately available):  `python setup.py develop` (you'll probably need to be administrator)
3) You're ready to go: try running the examples in the `example` folder.


`(c) Mark W. Mueller 2014`

 


