# Simulation tool of a programmable self-assembly in a robot swarm
Based on https://science.sciencemag.org/content/345/6198/795, main
article and supplementary material

### Required packages:
- mesa
- numpy
- matplotlib
- time
- pandas
- itertools
- scipy

They can be installed using pip

### Launch
1. Configure the simulation paramter in `parameters.py`
2. Make sure the log folder is created if runnning
   `test_edge_following`, `test_rectangle`, or `test_localize` from
   `main.py`
3. Launch `main.py` to launch a simulation run. The log file, if any, is
   registered at the end of the run
4. Modify the filename in the `data_analysis.py` file
5. Launch the `data_analysis` to analyse the results of the experiment.
6. Example of run are provided.

Details of model and implementation are in the related report. If any
question, do not hesitate to contact.


