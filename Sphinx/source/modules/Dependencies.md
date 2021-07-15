
# Dependencies


There are multiple dependencies that need to be met before installing and running the ``GME`` software (i.e., the ``gme`` Python package and its related Jupyter notebooks).

The basic requirement is to have ``Python 3.8`` (or later) installed. Most of ``GME`` development has been with 3.8, and current testing is against this version. While we have resisted using recent innovations (such as the walrus operator), problems may arise using any older version of Python.

The ``gme`` package relies heavily on ``NumPy``, ``SymPy``, ``SciPy``, and ``Matplotlib``, and several other Python packages. We list below the vintages of these packages guaranteed to work with ``GME``; newer versions may deprecate functionality that ``GME`` depends on.

The 'geomorphysics library' package [``gmplib``](https://github.com/cstarkjp/GMPLib/tree/main/Packages/gmplib) is also needed by ``gme``: it provides a set of utility functions for JSON parameter file parsing, general file I/O, and graphics.
Dependency on the ``gmplib`` package leads to a consequent dependency on the Python packages ``PIL`` (Pillow), ``json``, and ``IPython``.

Release 1.0 of ``GME`` has been developed with and tested against the following:

```eval_rst
.. list-table::
   :widths: 25 10 25
   :header-rows: 1

   * - Python package
     - Version
     - Needed by
   * - NumPy
     - 1.19.4
     - gme, Jupyter notebooks
   * - SymPy
     - 1.6.1
     - gme, Jupyter notebooks
   * - SciPy
     - 1.5.0
     - gme
   * - Matplotlib
     - 3.2.2
     - gme
   * - gmplib
     - 1.0
     - gme, Jupyter notebooks
   * - json
     - 2.0.9
     - gmplib
   * - PIL
     - 7.2.0
     - gmplib
   * - IPython
     - 7.15.0
     - gmplib, Jupyter notebooks
   * - Jupyter core
     - 4.6.3
     - Jupyter notebooks
   * - jupyter-notebook
     - 6.0.3
     - Jupyter notebooks
   * - nbconvert
     - 5.6.1
     - batch execution of Jupyter notebooks
   * - jupyter_contrib_nbextensions
     - 0.4.1
     - Jupyter notebooks (optional)
```

Of course, each of these Python packages brings with it another set of dependencies, most of which will be automatically met if installation is performed with ``pip``.
