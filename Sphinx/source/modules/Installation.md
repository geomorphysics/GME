
# Installation and set-up

The notes here are written for MacOS users, but will likely work with little to no modification on Linux.

Steps:

1) The best way to ensure a robust installation and runtime environment is to create a virtual environment and use it to install a fresh version of Python and the rest from scratch. While this sounds onerous, it can often be a time saver in the long run.

2) Using ``pip``, make sure all the Python, IPython and Jupyter
[dependencies](Dependencies.md)  are met.

3) Choose a parent directory for both ``GMPLib`` and ``GME`` and enter that directory. Here we will assume this directory is located in ``${HOME}/Projects``

4) Clone both of the repositories into this directory:
    ```
    git clone https://github.com/cstarkjp/GMPLib.git
    git clone https://github.com/cstarkjp/GME.git
    ```

5) Add the following environment variables to your ``.bash_profile`` or equivalent, modifying to match your choice of parent directory location:

    ```
    export GMPLHOME="${HOME}/Projects/GMPLib"
    export GMEHOME="${HOME}/Projects/GME"
    export PATH="${PATH}:${GMPLHOME}/Packages/gmplib:${GMEHOME}/Packages/gme"
    export PYTHONPATH="${PYTHONPATH}:${GMPLHOME}/Packages:${GMEHOME}/Packages"
    ```

    To enact these additions, either source this amended file or launch a new shell.
