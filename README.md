# narps_meta

This repository holds the code that I used for a research project for a course, Neuroimaging Data Science. Essentially, we looked at analytic variability - how methods influence results - in brain imaging. 

## Abstract
### Background 
Analysis choices may influence results in research using functional magnetic resonance imaging (fMRI). Studies have shown this using simulations on real data [1]. But until recently, it was unclear how much analytic variability there is in the field, between actual research groups.

> ### Purpose
> 1) **Assessing the problem.** Here, we complement the work by Botvinik-Nezer and colleagues [2], demonstrating the use of a multivariate analysis for investigating analytic variability. 
> 2) **Exploring a solution.** Further, we explore a novel method for exploring whether the meta-analytic result could be a potential solution to analytic variability.

### Data 
We used the data from the [Neuroimaging Analysis Replication and Prediction Study](https://www.narps.info/) (NARPS) [2]. In that study, 70 analysis teams analyzed the same task-fMRI dataset, performing whole-brain corrected analyses of task-related activation.

In the present study, our data consisted of the **group-level z-maps from 55 analysis teams**. We used the maps from one condition for the main analysis, and the maps from the other condition for replication.

### Methods
1) **Assessing the problem.** We used partial least squares to assess the association between methodological choices and the whole-brain group-level results. We used permutation testing to assess the significance of each component, and within significant component(s), we used bootstrap resampling with replacement to test the stability of each analysis variable’s contribution to the component. We followed up by comparing the analytic range of maps that did and did not use the provided preprocessed data.
2) **Exploring a solution.** Next, we explore whether the consensus result across teams was more accurate than the individual group-level maps, with ‘accuracy’ defined as similarity (correlation) to the consensus map from the literature. We used a fixed-effects general linear model to find the consensus across teams. We used [NeuroQuery](https://neuroquery.org/) [3] to find the consensus from the literature. We used Spearman correlation to measure the similarity between the two consensus maps, and spin permutation [4] to test for significance while accounting for spatial autocorrelation. 
3) **Replication.** We replicated the previous two analysis steps using maps from the same analysis teams, but these maps were group-level analyses of a different condition, involving a separate group of participants. 


### Results & Discussion
Our multivariate analysis for investigating analytic variability suggests that the more important analysis choices were 
- Whether the team used the fmriprep-preprocessed data, 
- The size of their smoothing kernel, and 
- Whether they used randomise to produce their statistical images (in FSL). 

Note that in the replication of these analyses in the second task condition (with separate participants), the smoothing kernel did not make a stable contribution to the first component. 

Notably, while the original publication using this dataset [2] also found that variability in results was related to the smoothing kernel and some choices of analysis software, they reported no significant effect for the use of the provided preprocessed data versus performing custom preprocessing. 

Our evaluation of the meta-analytic result as a potential solution to analytic variability did not show that the consensus across teams was more ‘accurate’ than the results from individual teams. However, our use of the consensus result from the literature as the ground-truth for evaluating accuracy may not be ideal. 

**References**
- [1] Carp, J. (2012). On the plurality of (methodological) worlds: estimating the analytic flexibility of FMRI experiments. Frontiers in neuroscience, 6, 149.
- [2] Botvinik-Nezer, R., Holzmeister, F., Camerer, C.F., Dreber, A., Huber, J., Johannesson, M., Kirchler, M., Iwanir, R., Mumford, J.A., ..., Nichols, T.E., Poldrack, R.A., Schonberg, T. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. Nature. https://doi.org/10.1038/s41586-020-2314-9. 
- [3] Dockès, J., Poldrack, R. A., Primet, R., Gözükan, H., Yarkoni, T., Suchanek, F., ... & Varoquaux, G. (2020). NeuroQuery, comprehensive meta-analysis of human brain mapping. Elife, 9, e53385.
- [4] Alexander-Bloch, A. F., Shou, H., Liu, S., Satterthwaite, T. D., Glahn, D. C., Shinohara, R. T., ... & Raznahan, A. (2018). On testing for spatial correspondence between maps of human brain structure and function. Neuroimage, 178, 540-551.

## Installation

To re-run these analyses, (*note that this won't work until the data is released publically.*)
1. Download the dataset and move it to this directory: `./data_narps/`
2. Get this repository: `$ git clone https://github.com/koudyk/narps_meta.git`
3. Go to the repo: `$ cd narps_meta`
4. Build the docker image: `$ docker build -t narps_meta_analysis .`
5. Start the docker container: `$ ./start_docker.sh`
6. Copy and paste the link from your terminal into a web browser to access the notebook.
7. In the Jupyter notebook environment, click on the "analyses" folder.  
8. Open a notebook (.ipynb file) and click "Cell >> Run Cells" from the menu at the top of the page.

Note that some of this code was borrowed/adapted from:
- [Alexandre Perez](https://github.com/alexprz/meta_analysis_notebook),
- [Ross Markello](https://netneurotools.readthedocs.io/en/latest/auto_examples/plot_mirchi_2018.html), and
- [BrainSpace](https://brainspace.readthedocs.io/en/development/generated/brainspace.null_models.spin.SpinPermutations.html) 
