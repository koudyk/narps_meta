#!/bin/sh

set -e

# Generate Dockerfile or Singularity recipe.
generate() {
  docker run --rm kaczmarj/neurodocker:master generate "$1" \
    --base=neurodebian:stretch-non-free \
    --pkg-manager=apt \
    --install fsl gedit python3 git gcc g++\
    --add-to-entrypoint='source /etc/fsl/fsl.sh' \
    --miniconda \
      conda_install="python=3.6
                      joblib
                      nilearn
                      numpy
                      scipy==1.2.0
                      jupyter" \
      pip_install="nilearn
            nistats
            statsmodels==0.10.0rc2
            duecredit
            sympy
            nipy
            git+https://github.com/alexprz/NiMARE.git@fdr-corrector" \
      create_env="neuro_py36" \
      activate=true \
    --run-bash "source activate neuro_py36 && git clone https://github.com/alexprz/nipy.git && cd nipy && python setup.py install" \
    --user=neuro \
    --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
    --workdir /home/neuro \
    --cmd jupyter notebook
}

generate docker > Dockerfile

docker build -t narps_meta_analysis .
