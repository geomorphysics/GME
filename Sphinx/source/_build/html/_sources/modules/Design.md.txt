
# GME software design

The GME software is designed around three elements:

1) The ``gme`` and ``gmplib`` Python packages

2) Jupyter/IPython notebooks

3) JSON parameter files

Each of these elements is described in more detail below.

## 1) Python packages

The main ``GME`` project package ``gme`` facilitates the following:

  - [CAS](https://en.wikipedia.org/wiki/Computer_algebra_system) (SymPy) solution of the geomorphic Hamiltonian and related equations 

  - integration of systems of 1st order ODEs (Hamilton's equations) for ray tracing

  - post-processing of rays to resolve topographic surface isochrones, knickpoints

  - visualization of rays, isochrones, and a variety of analyses of the results

The counterpart geomorphysics library package ``gmplib`` provides basic utility functions for graph plotting, JSON parameter file parsing, and file output. It is split from the ``gme`` package so that it can be used comfortably by other Python-based projects.


## 2) Jupyter/IPython notebooks

Jupyter notebooks are used to organize a sequence of processing and visualization tasks into a single file. The majority of the notebooks load and parse a parameter file (or several), solve the set of GME equations, execute a particular numerical integration (such as the tracing of a single ray), carry out post-processing (such as construction of a time-invariant surface from a single ray), visualize these results, and write the graphics to files.

Wrapper shell scripts make it possible to do bulk processing of multiple notebook jobs. The notebooks themselves are written in such a way that a sequence of parameter files can be specified externally by one of these shell scripts and passed in turn to the notebook in order to run each job in turn. Currently these wrapper scripts induce the notebook to successively overwrite itself upon completion of each job, but a simple modification would allow each notebook execution to be saved as a separate, appropriately named, file. This would allow in-situ examination of the results of each job.


## 3) JSON parameter files

Each GME processing job is controlled by a small set of JSON parameter files.
A JSON file is a convenient means of communicating the information required to formulate the full GME equation set (e.g., by specifying eta and mu) and to set up a numerical solution (e.g., by specifying erosion rate parameters, domain size, processing resolution, visualization parameters, etc).  A ``gmplib`` utility converts the combined JSON files into a Python dictionary containing the desired parameters, and this dictionary is mapped into the data properties of a parameters object (class instance)

At minimum, a notebook will parse a defaults JSON file and then a single "job" JSON file; some notebooks combine several job files.
