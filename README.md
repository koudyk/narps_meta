# narps_meta
Meta-analysis of the group-level statistical maps from the analysis teams involved in the NARPS project. This project is for a cours in Neuroimaging Data Science at McGill University (NEUR608, taught by Drs. Boris Bernhardt and Bratislav Misic)

To re-run these analyses, (*note that this won't work until the data is released publically.*)
1. Get this repository: `$ git clone https://github.com/koudyk/narps_meta.git`
2. Go to the repo: `$ cd narps_meta`
3. Download the dataset and move it to the this directory: `./data_narps/`
4. Generate the Dockerfile and build the docker image: `$ ./generate.sh`
5. Start the docker container: `$ docker run -it --rm -p 8888:8888 -v /path/to/current/directory/:/home/neuro/test/ narps_meta_analysis`
6. In the Jupyter notebook environment, click on the "analyses" folder.  
7. Open a notebook (.ipynb file) and click "Cell >> Run Cells" from the menu at the top of the page.

Note that much of this code was borrowed/adapted from Alexandre Perez (https://github.com/alexprz/meta_analysis_notebook).
