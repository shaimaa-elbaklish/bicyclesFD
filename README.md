# Bicycle Fundamental Diagram (BFD) method

## Data Sources
- **CRB Dataset**: Available at the GitHub repository:
https://github.com/DerKevinRiehl/mass_cycling_experiment.git
- **SRF Dataset**: Available at the GitHub repository: https://github.com/DerKevinRiehl/trajectory_analysis.git
- **NGSIM Dataset**: Available at https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

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
The Bicycle Fundamental Diagram: Empirical Insights into Bicycle Flow for Sustainable Urban Mobility (2025).
El-Baklish, S.K. and Riehl, K. and Ni, Y-C. and Ramseier, T. and Kouvelas, A. and Makridis, M.
(Submitted to Scientific Reports)
```
```
Mass-Cycling Experiment Trajectory Dataset (2025).
Riehl, K. and El-Baklish, S.K. and Ni, Y-C. and Ramseier, T. and Kouvelas, A. and Makridis, M.
(Submitted to Data in Brief)
```
```
The Fundamental diagram of autonomous vehicles: Traffic state estimation and evidence from vehicle trajectories (2025).
Makridis, M. and El-Baklish, S.K. and Kouvelas, A. and Laval, J.
(Accepted in Communications in Transportation Research)
```
