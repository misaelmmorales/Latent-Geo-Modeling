mrstModule add ad-core ad-blackoil ad-props spe10 mrst-gui

G=cartGrid([48,48,8], [1000,1000,80]*meter); G=computeGeometry(G);

load facies_maps_48_48_8.mat
load logperm_48_48_8.mat
clear Label

fluid = initSimpleADIFluid('phases',    'WO', ...
                           'mu',        [1,    5]*centi*poise, ...
                           'rho',       [1000, 850], ...
                           'n',         [2,    2], ...
                           'c',         [0,    1e-5]/barsa, ...
                           'cR',        4e-10/barsa);

n_realizations = size(TI,1);

parfor i=1:n_realizations
    run_simulation_waterdrive_3d(i, G, fluid, perm, TI);
end