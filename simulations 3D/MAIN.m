%% Initialize MRST
clear; close all; clc
mrstModule add incomp spe10 
linsolve = @mldivide;
gravity on
set(0,'DefaultFigureWindowStyle','docked')

realization = 19;

%% Grid generation
G  = cartGrid([128 128 16], [1000 1000 100]*meter);
G  = computeGeometry(G);

%% Rock and Fluid properties
perm_all = csvread('perm_all.csv', 1);
channel_all = csvread('channel_all.csv');

channel = reshape(reshape(channel_all(:,realization), [128,128])', [], 1);
channel(channel==0) = 0.8;

poro = channel.*10.^((perm_all(:,realization)-7)/10);
perm = channel.*(10.^perm_all(:,realization))*milli*darcy;
permeability = convertTo(perm, milli*darcy); %for export

rock = makeRock(G, perm, poro);

fluid = initSimpleFluid('mu',  [1, 5]*centi*poise, ...
                        'rho', [1000, 850]*kilogram/meter^3, ...
                        'n'  , [2, 2]);

%% Visualization Por/Perm
figure(1)
plotCellData(G, rock.poro, 'EdgeAlpha', 0.2);
title('Porosity [v/v]'); colorbar; colormap('jet'); view(-10,85)

figure(2)
plotCellData(G, log10(convertTo(rock.perm,milli*darcy)), 'EdgeAlpha', 0.2);
title('log_{10} Permeability [mD]'); colorbar; colormap('jet'); view(-10,85)

figure(3)
plotCellData(G, channel); colormap('gray')
title('Fluvial Channels'); colorbar; view(-10,85)

%% Wells
I = [1 1 1];
J = [1 64 128];
R = [500 500 500]*stb/day;    
nIW = 1:numel(I); W = [];
for i = 1 : numel(I)
    W = verticalWell(W, G, rock, I(i), J(i), [], ...
        'Type', 'rate', ...
        'lims', 1E4*psia, ...
        'InnerProduct', 'ip_tpf', ...
        'Val', R(i), ...
        'Radius', 0.1, ...
        'Comp_i', [1, 0], ...
        'name', ['I', int2str(i)]);
end
NumInj=length(I);

I = [128 128 128];
J = [1 64 128];
P = [2000 2000 2000]*psia;
nPW = (1:numel(I))+max(nIW);
for i = 1 : numel(I)
    W = verticalWell(W, G, rock, I(i), J(i), [], ...
        'Type', 'bhp', ...
        'lims', 1E4*psia, ...
        'InnerProduct', 'ip_tpf', ...
        'Val', P(i), ...
        'Radius', 0.1, ...
        'name', ['P', int2str(i)], ...
        'Comp_i', [0, 1]);
end

%% Transmissibilities and initial state
trans = computeTrans(G, rock);
rSol  = initState(G, W, 3000*psia, [0.2,0.8]);
rSol = incompTPFA(rSol, G, trans, fluid, 'wells', W, 'LinSolve', linsolve);

%% Main loop
T = 15*year;     %total time
dT = 1*year/3;   %time step
total_timesteps = T/dT;
dTplot = dT;

% Prepare plotting of saturations
figure(4)
movegui(figure(4));
h=colorbar; colormap jet; view(-10,85)
[hs_p,ha_p] = deal([]);

%% Start the main loop
t  = 0;  plotNo = 1;
wres = cell([1, 4]);

Prod = struct('t'  , []                  , ...
              'vpt', zeros([0, numel(W)]), ...
              'opr', zeros([0, numel(W)]), ...
              'wpr', zeros([0, numel(W)]), ...
              'wc' , zeros([0, numel(W)]));
Satu = zeros(G.cartDims(1)*G.cartDims(2), total_timesteps);
Pres = zeros(G.cartDims(1)*G.cartDims(2), total_timesteps);

append_wres = @(x, t, vpt, opr, wpr, wc) ...
   struct('t'  , [x.t  ; t                  ], ...
          'vpt', [x.vpt; reshape(vpt, 1, [])], ...
          'opr', [x.opr; reshape(opr, 1, [])], ...
          'wpr', [x.wpr; reshape(wpr, 1, [])], ...
          'wc' , [x.wc ; reshape(wc , 1, [])]);

[wres{:}] = prodCurves(W, rSol, fluid);
Prod      = append_wres(Prod, t, wres{:});

while t < T
    rSol = implicitTransport(rSol, G, dT, rock, fluid, 'wells', W);
    assert(max(rSol.s(:,1)) < 1+eps && min(rSol.s(:,1)) > -eps);
    rSol = incompTPFA(rSol, G, trans, fluid, 'wells', W);
    t = t + dT;

    if ( t < plotNo*dTplot && t <T), continue, end
    % Plot saturation
    figure(4)
    delete([hs_p, ha_p])
    hs_p = plotCellData(G, rSol.s(:,2), 'EdgeAlpha', 0.2);
    plotWell(G, W, 'height', 10, 'color', 'k'); view(-10,85)
    ha_p = annotation('textbox', [0 0.93 0.32 0.07], ...
                      'String', ['Oil Sat at ', ...
                      num2str(convertTo(t,year)), ' years'], 'FontSize',8);
   drawnow
   [wres{:}] = prodCurves(W, rSol, fluid);
   Prod      = append_wres(Prod, t, wres{:});
   Satu(:, plotNo) = rSol.s(:,2);
   Pres(:, plotNo) = convertTo(rSol.pressure, psia);
   plotNo = plotNo+1;
end

%% Save results
prod_name = sprintf("response_production/production_%d.mat", realization);
satu_name = sprintf("response_saturation/saturation_%d.mat", realization);
pres_name = sprintf("response_pressure/pressure_%d.mat", realization);
poro_name = sprintf("features_porosity/porosity_%d.mat", realization);
perm_name = sprintf("features_permeability/permeability_%d.mat", realization);

save(fullfile(pwd(), prod_name), 'Prod')
save(fullfile(pwd(), satu_name), 'Satu')
save(fullfile(pwd(), pres_name), 'Pres')
save(fullfile(pwd(), poro_name), 'poro')
save(fullfile(pwd(), perm_name), 'permeability')

%% END