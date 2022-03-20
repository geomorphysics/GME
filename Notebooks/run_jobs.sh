#!/usr/bin/env bash

if [[ $1 =~ ^[-]+[h]*$ ]] || [[ $2 =~ ^[-]+[h]*$ ]]; then
    printf "\nrun_nb.sh [<notebook_name>] [-h/--help]\n\n"
    printf "Parses <notebook_name>_jobs.py and runs through its list of GME parameter file name(s),\n"
    printf "assigning each in turn to an environment variable and then executing the Jupyter/IPython\n"
    printf "<notebook_name>.ipynb.\n\n"
    exit
fi

# if [[ $1 =~ ^[-]+[t]*$ ]]; then
#     echo "Testing! \$1"
#     nb_filename="Test.ipynb"
#     jobs_filename=$2"_jobs.py"
#     p=`echo "$(cd "$(dirname "$2")"; cd ../..; pwd -P)"`
# elif [[ $2 =~ ^[-]+[t]*$ ]]; then
#     echo "Testing! \$2"
#     nb_filename="Test.ipynb"
#     jobs_filename=$1"_jobs.py"
#     p=`echo "$(cd "$(dirname "$1")"; cd ../..; pwd -P)"`
# else
nb_filename=$1".ipynb"
jobs_filename=$1"_jobs.py"
p=`echo "$(cd "$(dirname "$1")"; cd ../..; pwd -P)"`
# fi
export GME_WORKING_PATH=$p

echo
echo "Setting working path to \"$GME_WORKING_PATH\":"
echo "Parsing \"$jobs_filename\":"
echo

while IFS= read -r line; do
    if [[ $line =~ ^[^#]+[^\n]*$ ]]; then
        echo "$line"
    fi
done < $jobs_filename

while IFS= read -r line; do
    echo
    if [[ $line =~ ^[^#]+[^\n]*$ ]]; then
        echo; echo
        export GME_NB_PR=$line
        echo "Run \"$nb_filename\" on $line"
        echo
        jupyter nbconvert --to notebook --execute $nb_filename \
                    --log-level=40 --ExecutePreprocessor.timeout=-1 --clear-output
    else
        echo "     \"$line\""
        echo
    fi
done < $jobs_filename
