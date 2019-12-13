# narps_meta
Meta-analysis of the group-level statistical maps from the analysis teams involved in the NARPS project. This project is for a graduate course in Neuroimaging Data Science at McGill University (NEUR608, taught by Drs. Boris Bernhardt and Bratislav Misic).

To re-run these analyses, (*note that this won't work until the data is released publically.*)
1. Download the dataset and move it to this directory: `./data_narps/`
2. Install [docker](https://docs.docker.com/v17.09/engine/installation/)
3. Get this repository: `$ git clone https://github.com/koudyk/narps_meta.git`
4. Go to the repo: `$ cd narps_meta`
5. Get a [Freesurfer licence](https://surfer.nmr.mgh.harvard.edu/registration.html), make a folder called `freesurfer_licence` in the current directory, and put the `licence.txt` file that you get from Freesurfer into that folder
5. Build the docker image: `$ docker build -t narps_meta_analysis .`
6. Start the docker container: `$ docker run -it --rm -p 8888:8888 -v /path/to/current/directory/:/home/neuro/test/ narps_meta_analysis`
7. Copy and paste the link from your terminal into a web browser to access the notebook.
8. In the Jupyter notebook environment, click on the "analyses" folder.  
9. Open a notebook (.ipynb file) and click "Cell >> Run Cells" from the menu at the top of the page.

Note that some of this code was borrowed/adapted from:
- Alexandre Perez (https://github.com/alexprz/meta_analysis_notebook),
- Neu
