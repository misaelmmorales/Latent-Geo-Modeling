%% Run SNESIM from mGSTAT for random fluvial channel realizations
% This script is really slow for a large number of realizations. Run once.
% Specify:
% nx, ny, nz = number of cells in x-, y-, and z- directions
% nsim = number of realizations to simulate

[nx, ny, nz] = deal(128, 128, 16);
numsim = 6;

S = sgems_get_par('filtersim_cont');
%S = sgems_get_par('snesim_std');
S.ti_file = 'snesim_std.ti';
S.dim.nx = nx;
S.dim.ny = ny;
S.dim.nz = nz;
S.XML.parameters.Nb_Realizations.value = numsim;
S.XML.parameters.Scan_Template = [11,11,11];
S.XML.parameters.Nb_Multigrids_ADVANCED = 5;
S.XML.parameters.Data_Weights = [0.4, 0.4, 0.2];
S = sgems_grid(S);

csvwrite('channels_3d.csv', S.data);

G=cartGrid([nx,ny,nz],[1000,1000,100]); G=computeGeometry(G);
for i=1:6
    subplot(2,3,i)
    plotCellData(G, S.data(:,i)); view(-10,85);
end