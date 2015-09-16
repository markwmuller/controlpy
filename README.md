Controlpy
=========

A library for commonly used controls algorithms (e.g. creating LQR controllers). An alternative to Richard Murray's "control" package -- however, here we do not require Slycot.

Current capabilities:

1. System analysis:
	1. Test whether a system is stable, controllable, stabilisable, observable, or stabilisable.
	2. Get the uncontrollable/unobservable modes
	3. Compute a system's controllability Gramian (finite horizon, and infinite horizon)
	4. Compute a system's H2 and Hinfinity norm
2. Synthesis
	1. Create continuous and discrete time LQR controllers
	2. Full-information H2 optimal controller
	3. H2 optimal observer
	4. Full-information Hinf controller


How to install
--------------
Install using pypi, or direct from the Github repository:

1. Clone this repository somewhere convenient: `git clone https://github.com/markwmuller/controlpy.git`
2. Install the package (we'll do a "develop" install, so any changes are immediately available):  `python setup.py develop` (you'll probably need to be administrator)
3. You're ready to go: try running the examples in the `example` folder.



Licensing
---------
`(c) Mark W. Mueller 2015`

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

 


