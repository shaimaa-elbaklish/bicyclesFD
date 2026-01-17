# Bicycle Fundamental Diagram (BFD) method

## Data Sources
- **BikeZ-ETH-RA Dataset**: Available at the GitHub repository:
https://github.com/DerKevinRiehl/mass_cycling_experiment.git
- **SRF Dataset**: Available at the GitHub repository: https://github.com/DerKevinRiehl/trajectory_analysis.git
- **ARED Dataset**: Available at http://hdl.handle.net/1803/9358

## Installation & Run Instructions for Reproduction
Please create a dedicated conda environment (recommended Python 3.11+). To install requirements, please use the package management system pip as follows:
```
pip install -r requirements.txt
```

Please update the root paths for the data directories in the `src/_constants.py` file; specifically the data classes `CRB_Config` and `SRF_Config`.

To run the BFD method, the main code is
```
src/main_CRB_fd_analysis.py
```

To replicate the comparison across traffic modes (i.e. cars versus bicycles), please run
```
src/figure_7_df_comparison_cars.py
```
The PFD method (for motorized vehicles) is used to generate the traffic states for the NGSIM dataset. Please refer to the following GitHub repository: https://github.com/shaimaa-elbaklish/pfd_tse.git

## Publications
```
Quantifying Bicycle Flow Efficiency: Inference of the Fundamental Diagram from Experimental Observations (2026).
El-Baklish, S.K. and Riehl, K. and Ni, Y-C. and Ramseier, T. and Kouvelas, A. and Makridis, M.
(Submitted to Nature Communications)
```
```
BikeZ-ETH – A Mass-Cycling Trajectory Dataset from a Controlled Experiment (2026).
Riehl, K. and El-Baklish, S.K. and Ni, Y-C. and Kouvelas, A. and Makridis, M.
(Submitted to Scientific Data)
```
```
The Fundamental diagram of autonomous vehicles: Traffic state estimation and evidence from vehicle trajectories (2025).
Makridis, M. and El-Baklish, S.K. and Kouvelas, A. and Laval, J.
(Accepted in Communications in Transportation Research)
```
