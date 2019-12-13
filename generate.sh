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
                      jupyter
                      matplotlib
                      seaborn" \
      pip_install="git+git://github.com/nilearn/nilearn.git@2ead4d01df71d51c205a437a8fc5419ecb2a0beb
            nistats
            statsmodels==0.10.0rc2
            duecredit
            sympy
            nipy
            git+https://github.com/alexprz/NiMARE.git@fdr-corrector
            git+https://github.com/netneurolab/netneurotools.git@0f0b03de92ccdd170c3648ee59292b8c9f25e8d7
            git+https://github.com/MICA-MNI/BrainSpace.git" \
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


#--freesurfer version=6.0.0 mefthod=binaries \
#--copy freesurfer_licence/license.txt /opt/freesurfer-#6.0.0/ \
