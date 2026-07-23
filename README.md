Topological reconstruction of one-particle-thick stripes

This repository contains the simulation, analysis, and plotting scripts usedto study the thermal transformation of one-particle-thick stripes into finitepolymer-like clusters in a two-dimensional Lennard-Jones plus Gaussian(LJG) system.

The production protocol follows continuous isochoric heating and coolingpaths. The same configuration is propagated through all temperatures, so thethermal history is preserved. Structural, graph-topological, filament, anddynamical observables are obtained from the saved trajectories.

Repository layout

.
├── simulation/
│   ├── espresso_ljgauss_2d_thermal_cycle.py
│   └── run_ljgauss_2d_thermal_cycles.py
├── analysis/
│   ├── analyze_ljgauss_thermal_cycles.py
│   └── run_analysis.sh
├── figures/
│   ├── article_figure_01_structural_pathway.py
│   ├── article_figure_02_topological_reconstruction.py
│   ├── article_figure_03_chain_geometry.py
│   ├── article_figure_04_topology_dynamics.py
│   ├── article_figure_05_cluster_cycle_distributions.py
│   └── make_supplemental_figures.py
├── requirements-analysis.txt
└── README.md

Simulation outputs and generated figures are intentionally excluded from Gitthrough .gitignore.

Software requirements

The simulations require ESPResSo with itsPython interface and the Lennard-Jones and tabulated nonbonded interactionsenabled. The scripts were prepared for ESPResSo 4.3.x and use pypresso.

The structural analysis should be run in an ordinary Python environment,separate from ESPResSo if desired:

python3 -m venv analysis_env
source analysis_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt

Python 3.10 or newer is recommended.

Model and default production protocol

The pair potential is

U(r)=4\varepsilon\left[(\sigma/r)^{12}-(\sigma/r)^6\right]
 +A\varepsilon\exp\left[-\left(\frac{r-r_0\sigma}{c\sigma}\right)^2\right].

The default model parameters are:

epsilon = sigma = m = 1;

A = 5, r0 = 0.7, and c = 1;

interaction cutoff r_cut = 3.5;

time step dt = 0.002;

Langevin friction gamma = 1;

strict two-dimensional dynamics;

N = 4096, with a perfect initial stripe array generated using--stripe-nx 64;

densities rho = 0.395, 0.405, and 0.415;

four independent seeds;

heating from T = 0.04 to 0.20 in increments of 0.005, followed bycooling through the reversed temperature sequence;

10^6 equilibration steps at the first state point, 3 x 10^5 at eachsubsequent state point, and 5 x 10^5 production steps.

The native ESPResSo Lennard-Jones interaction is combined with a tabulatedGaussian contribution. This preserves the divergent repulsive core and avoidsextrapolating a tabulated potential into the particle-overlap region.

Running one state-point sequence

From the repository root:

pypresso simulation/espresso_ljgauss_2d_thermal_cycle.py \
  --N 4096 \
  --rho 0.405 \
  --seed 1 \
  --stripe-nx 64 \
  --outdir thermal_cycles/rho_0.40500/seed_1

This command runs the complete heating and cooling cycle for one density andone seed.

For a short installation test, reduce the run lengths:

pypresso simulation/espresso_ljgauss_2d_thermal_cycle.py \
  --N 256 \
  --rho 0.405 \
  --seed 1 \
  --stripe-nx 16 \
  --temperatures 0.04 0.05 \
  --steps-eq-first 1000 \
  --steps-eq 1000 \
  --steps-prod 2000 \
  --sample-every 100 \
  --thermo-every 100 \
  --traj-every 100 \
  --outdir thermal_cycles_test/rho_0.40500/seed_1

The short test is intended only to verify the installation and output layout;it does not reproduce the production results.

Running densities and seeds in parallel

Independent density/seed combinations can be launched concurrently:

python simulation/run_ljgauss_2d_thermal_cycles.py \
  --pypresso pypresso \
  --out-root thermal_cycles \
  --rhos 0.395 0.405 0.415 \
  --seeds 1 2 3 4 \
  --N 4096 \
  --stripe-nx 64 \
  --nproc 4

--nproc is the maximum number of independent ESPResSo processes running atthe same time. This is trajectory-level parallelism: one density/seed pair ishandled by each process. Increase it only when sufficient CPU cores and memoryare available.

The launcher skips a run when its cycle_summary.dat already exists. Use--force to rerun it.

Output structure

The simulation creates:

thermal_cycles/
└── rho_0.40500/
    └── seed_1/
        ├── metadata.json
        ├── cycle_summary.dat
        ├── heating/
        │   └── T_0.04000/
        │       ├── trajectory.gsd
        │       ├── thermo.dat
        │       ├── msd_vacf.dat
        │       └── config_final.npz
        └── cooling/
            └── T_0.20000/
                └── ...

GSD positions are wrapped into the primary periodic box. New trajectoriesalso store the ESPResSo particle image counters, allowing exact reconstructionof continuous coordinates.

Structural and topological analysis

Activate the analysis environment and run:

python analysis/analyze_ljgauss_thermal_cycles.py \
  --root thermal_cycles \
  --rhos 0.395 0.405 0.415 \
  --bond-cutoff 1.5 \
  --frame-stride 1 \
  --workers 8

The convenience wrapper contains the same production command:

bash analysis/run_analysis.sh

The graph analysis uses a default neighbor cutoff of 1.5 sigma. Cutoffrobustness is evaluated from 1.35 to 1.65.

Principal observables include:

global nematic order and local twofold bond order;

curvature and particle coordination;

connected components and periodic winding;

Betti numbers and Euler characteristic;

graph 2-core, fundamental cycles, triangles, and transitivity;

cluster-size and cluster-topology distributions;

contour length, end-to-end distance, tortuosity, tangent correlations, andpersistence length for finite unbranched chains;

bond survival and graph-resolved bond formation or rupture;

endpoint-, backbone-, and branch-conditioned displacement;

self-diffusion from the long-time two-dimensional MSD.

Corrected MSD calculation

D_msd in statepoints_by_seed.dat and ensemble_summary.dat is recomputedby the analysis script and should be used for the manuscript figures.

The procedure is:

reconstruct continuous particle trajectories from the stored GSD imagecounters;

for legacy GSD files without image counters, unwrap consecutive frameswith the minimum-image displacement;

evaluate the MSD using every available time origin;

retain lags up to one half of the trajectory;

fit the final half of that retained interval toMSD(t) = 4 D t + b.

The default fitting choices can be changed with:

--msd-max-lag-fraction 0.5 --msd-fit-fraction 0.5

For every state point, the analysis writes the fitted MSD curve and number oftime origins to:

thermal_cycles/analysis/per_seed/rho_*/seed_*/*/T_*_msd.dat

The columns msd_fit_R2, msd_fit_loglog_alpha, msd_fit_tau_min, andmsd_fit_tau_max permit direct inspection of fit quality. The oldersimulation-time estimate is retained as D_msd_inline only as a diagnostic.D_vacf is read from cycle_summary.dat.

The minimum-image fallback assumes that no particle moves by more than half abox length between consecutive saved frames. The production sampling intervalsatisfies this condition, but stored image counters are preferred.

Main analysis files

The analysis directory contains:

statepoints_by_seed.dat: time-averaged values for every independent seed;

ensemble_summary.dat: equal-weight mean and standard error over seeds;

cluster_size_distribution_by_seed.dat;

cluster_topology_distribution_by_seed.dat;

linear_chain_statistics_by_seed.dat;

tangent_correlation_by_seed.dat;

two_core_size_distribution_by_seed.dat;

fundamental_cycle_length_distribution_by_seed.dat;

betti_vs_cutoff_by_seed.dat;

orientational_correlation_by_seed.dat;

frame-resolved and MSD-resolved files below per_seed/.

The generated analysis/README.txt documents every output column and theweighting conventions used in the distributions.

Reproducing the figures

After the analysis has finished:

python figures/article_figure_01_structural_pathway.py
python figures/article_figure_02_topological_reconstruction.py
python figures/article_figure_03_chain_geometry.py
python figures/article_figure_04_topology_dynamics.py
python figures/article_figure_05_cluster_cycle_distributions.py
python figures/make_supplemental_figures.py

The main figures are written to article_figures/; Supplemental figures arewritten to supplemental_figures/. All figure scripts accept explicit inputand output paths through their command-line options, exceptmake_supplemental_figures.py, whose paths are defined near the beginning ofthe file.

Reproducibility notes

Heating and cooling are path-dependent finite-rate protocols; they shouldnot be interpreted automatically as equilibrium coexistence branches.

The four seeds are averaged with equal statistical weight.

Structural averages are first evaluated within each seed and then combinedacross seeds.

Individual fundamental-cycle lengths depend on the selected cycle basis,whereas the number of basis elements equals the invariant cycle rankbeta1.

Winding components are detected using integer periodic-image offsets, notonly by the size of the largest cluster.

The simulation aborts if the initial minimum particle separation is below0.8 sigma.

Citation

If these scripts are used in published work, please cite the associatedarticle. The full citation and DOI should be added here after publication.
