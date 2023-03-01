%% Run SNESIM from mGSTAT for random fluvial channel realizations
% This script is really slow for a large number of realizations. Run once.
% Specify:
% nx, ny, nz = number of cells in x-, y-, and z- directions
% nsim = number of realizations to simulate

[nx, ny, nz] = deal(128, 128, 16);
numsim = 20;

S = sgems_get_par('filtersim_cont');
S.ti_file = 'filtersim_cont_channels.ti';
S.dim.nx = nx;
S.dim.ny = ny;
S.dim.nz = nz;
S.XML.parameters.Nb_Realizations.value = numsim;
S = sgems_grid(S);

csvwrite('channels_3d.csv', S.data);