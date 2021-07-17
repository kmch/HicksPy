.. fullwavepy documentation master file, created by
   sphinx-quickstart on Sun May 24 16:50:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FullwavePy: FWI workflow in Jupyter
===================================

Getting started
---------------
The best way to start is to play around with example notebooks
in `examples/`.

Key features
------------
Calculating coefficients of Hicks sources / receivers.

Key concepts
------------
FWI is conceptually simple but usually requires a lot of data manipulation
and parameter tuning to make it work successfully. Hence a need for 
convenient, robust and extensible framework that automates the process and 
allows to focus on research questions rather than technicalities.

FullwavePy's project is something defined uniquely by FWI **input** files. 


E.g. a project has exactly one *Runfile* but can have multiple auxiliary, job-specific files (e.g. *Out.log*, *Run.pbs*, *JobInfo.txt*, etc.) 
that have job-file identifiers embedded in their names (actual names *Out0.log*, etc.). Obviously, **output files** are job-specific too, but currently they are **overwritten after each job-run** instead of being endowed with such ids (to save disk space). To preserve them, one needs to create a new project.


The basic building blocks of any project are files. 
They are abstract objects i.e.
they are not bound to any specific I/O implementation.

The binding needed is done under the hood

and the tricky part is to get 
all the necessary data regardless of the io implementation.
This can be challenging e.g. when no headers are associated
with seismic data. An alternative way of getting this data
has to be implemented. Or the user should be informed of the 
limited functionality.

Plotting of the data should be independent of the io.


Requirements
------------
Remember to add parent-path/fullwavepy to your PYTHONPATH.
(parent-path is not enough since we have:
parent-path/fullwavepy/fullwavepy/__init__.py).

If the autogen fails to find one of the modules unlike all 
the other modules, most likely there is a syntax error inside 
this module. Try to import it from python interpreter to to track
down the bug.

.. table:: Truth table for "not"
   :widths: auto

   =====  =====
     A    not A
   =====  =====
   False  True
   True   False
   =====  =====



.. autosummary::
   :toctree: modules
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
