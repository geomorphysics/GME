# Running GME notebooks

For the Jupyter/IPython notebooks to run successfully, the environment variable `PYTHONPATH` must include paths to the parent directories of both the `GME` and `GMPLib` Python packages. On MacOS and Linux platforms this is achieved by modifying the start-up script of your preferred shell (see [notes on installation and set-up](Installation.md)) and by launching the Jupyter notebook server from this shell and accessing the server in a browser. The chosen `GME` notebook can then be opened and executed inline.

This approach is fine for single-shot runs of `GME` with a particular choice of parameters. However, if we want to run a group of analyses with several choices of parameter sets, inline execution is inconvenient: bulk, offline execution is much more efficient. Shell scripts (written in `bash`) are provided to make this possible.

For example, the [`TimeInvariant.ipynb`](Notebooks/RayTracing/TimeInvariant.ipynb) notebook can be loaded into a Jupyter notebook server, modified to work with a particular choice of JSON parameter file [selected from here](Parameters/RayTracing), and run inline in a browser. On the other hand, if we wish to run this notebook several times with against a set of parameter files, we can do the following:
  - specify the list of JSON parameter files in [`TimeInvariant_jobs.py`](Notebooks/RayTracing/TimeInvariant_jobs.py)
  - run the batch shell script [`run_jobs.sh`](Notebooks/run_jobs.sh) from the [notebook directory](Notebooks/RayTracing) using the shell command `../run_jobs.sh TimeInvariant_jobs` (with the path to the script modified as appropriate)

The script steps through the list of parameter files and sets the following environment variables
  - `GME_NB_PR`, short for "`GME` notebook parameter files" = Python list of parameter filename strings
  - `GME_WORKING_PATH` = the absolute path to the `GME` package (its root directory)

before invoking the Jupyter interpreter with `nbconvert` as follows:

    jupyter nbconvert --to notebook --execute $nb_filename \
            --log-level=40 --ExecutePreprocessor.timeout=-1 --clear-output

See [here](https://nbconvert.readthedocs.io/en/latest/execute_api.html) and [here](https://ipython.org/ipython-doc/3/notebook/nbconvert.html)  for more information on this technique.  

The [`run_jobs.sh`](Notebooks/run_jobs.sh) script as written runs the notebooks in-place, overwriting them each time execution completes. If you wish to record each executed notebook elsewhere, modify the invocation above to something like

    jupyter nbconvert --to notebook --execute $nb_filename --output $job_nb_filename \
            --log-level=40 --ExecutePreprocessor.timeout=-1 --clear-output

and in the loop reassign `$job_nb_filename` for each job.
