This repository contains code for reproducing results in the associated manuscript: "**Kinematic coupling of the glenohumeral and scapulothoracic joints generates humeral axial rotation**".

#### Installation Instructions

This repository relies on [Anaconda](https://www.anaconda.com/products/individual) for installing dependencies. All commands below should be run from the Anaconda Prompt. After installing Anaconda and cloning the repository, make sure you are in the same directory as `environment.yml` and run:

`conda env create --file environment.yml`

#### Configuration

All code within the repository relies on two configuration files (`logging.ini` and `parameters.json`) to locate the associated data repository, configure analysis parameters, and instantiate logging. The location of the `config` directory is specified as a command line parameter (so it is feasible for this folder to reside anywhere in the filesystem). Analysis and figure generation tasks must be executed as module scripts:

`python -m st_generated_axial_rot.figures.axial_rot_corr config`

Template `logging - template.ini` and `parameters - template.json` files are located in the `config` directory and should be copied and renamed to `logging.ini` and `parameters.json`. [Parameters.md](Parameters.md) explains each parameter within `parameters.json`. Each analysis and figure generation task contains Python documentation describing its utility and the parameters that it expects from `parameters.json`.

#### Supporting dataset and data

The associated data repository containing biplane fluoroscopy derived humerus and scapula kinematics can be found on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4536684.svg)](https://doi.org/10.5281/zenodo.4536684). The `biplane_vicon_db_dir` parameter should point to the `database` folder of this repository. This analysis also relies on 2 additional supporting data files which reside in the `data` directory.

#### Organization

The `config` directory contains configuration files.

The `data` directory contains two data files indicating the start and end frames that should be utilized when analyzing external rotation at 90&deg; of abduction and external rotation in adduction trials.

The `euler_angles_angular_velocity.mlx` file is a MATLAB Live Script that was utilized to generate Appendix 2 of the manuscript. It explains why Euler angles are ill-suited for quantifying physiologic joint motion and why angular velocity (once properly projected) is best suited for this task.

The `st_generated_axial_rot` directory contains code for reproducing the analysis of the associated manuscript. Within `st_generated_axial_rot` the following packages exist:

* `analysis` - contains code for various analyses that were undertaken to analyze the contributions of the scapulothoracic and glenohumeral joints to humerothoracic axial rotation, elevation, and plane of elevation.
* `figures` - contains code for reproducing the figures in the associated manuscript.
* `common` - contains common code utilized by both `analysis` and `figures`.

#### License

This code is licensed according to the most restrictive license ([GPLv3](https://choosealicense.com/licenses/gpl-3.0/)) of the packages that it utilizes: [spm1d](https://github.com/0todd0000/spm1d).
