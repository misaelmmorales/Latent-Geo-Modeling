%% Run SNESIM from mGSTAT for random fluvial channel realizations
% This script is really slow for a large number of realizations. Run once.
% Specify:
% nx, ny, nz = number of cells in x-, y-, and z- directions
% nsim = number of realizations to simulate

[nx, ny, nz] = deal(128, 128, 16);
numsim = 4;

S = sgems_get_par('filtersim_cont');
%S = sgems_get_par('snesim_std');
S.ti_file = 'snesim_std.ti';
S.dim.nx = nx;
S.dim.ny = ny;
S.dim.nz = nz;
S.XML.parameters.Nb_Realizations.value = numsim;
S = sgems_grid(S);

channels_3d = reshape(S.D, [], numsim);
csvwrite('channels_3d.csv', channels_3d);