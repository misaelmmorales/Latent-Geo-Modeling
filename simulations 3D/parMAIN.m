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

%% Plotter
%{
realization = 10;
load(['E:/Latent_Geo_Inversion/simulations_3D/saturation/saturation_',num2str(realization),'.mat'])
load(['E:/Latent_Geo_Inversion/simulations_3D/porosity/porosity_',num2str(realization),'.mat'])
load(['E:/Latent_Geo_Inversion/simulations_3D/permeability/permeability_',num2str(realization),'.mat'])
figure
subplot(1,3,1)
plotCellData(G, porosity); colormap jet; cb=colorbar; view(-20,75); cb.Label.String='Porosity [v/v]';
title(['Realization ', num2str(realization), ' Porosity'])
subplot(1,3,2)
plotCellData(G, np.log10(perm_md)); colormap jet; cb=colorbar; view(-20,75); cb.Label.String='Log-Permeability [log(mD)]';
title(['Realization ', num2str(realization), ' Permeability'])
subplot(1,3,3)
for i=1:40
sol = reshape(satu(i,:),[],1);
gca;
plotCellData(G, sol); view(-20,75); colormap jet; cb=colorbar; caxis([0,1]); cb.Label.String='Saturation [fraction]';
title(['Realization ',num2str(realization), ' Saturation'])
pause(.2)
end
fprintf('End\n')
%}