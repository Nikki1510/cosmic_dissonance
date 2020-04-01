# cosmic_dissonance
This repository contains the code used to generate the results from the paper "Cosmic dissonance: new physics or systematics behind a short sound horizon?" by Nikki Arendse, Radek Wojtak and Adriano Agnello (https://arxiv.org/pdf/1909.07986.pdf). 

### Description
In our paper, we use late-time probes such as supernova (SN) and Baryon Acoustic Oscillations (BAO) and calibrate them with absolute distance measurements from gravitational lensing, Cepheids and the Tip of the Red Giant Branch to obtain a measurement of the sound horizon and the Hubble constant. We use 4 different models to infer the expansion history: models 1-3 are polynomial parametrizations and do not depend on a cosmological model, and model 4 is the standard LCDM model.

### Usage
Download the complete cosmic_dissonance repository. With Python, run either
* _Run.py:_ to generate data. This will create new files in the folders Chains, Output and Testing.
* _Fig_test.py:_ to visually inspect the data. This will generate figures in the folder Figures. _Fig_test.py_ can only be ran for data that is already available. To begin with, I already provided an example data file (where model 3 is ran with SN, BAO and Cepheids).

### Contact
Any questions? Email me at nikki.arendse@nbi.ku.dk
