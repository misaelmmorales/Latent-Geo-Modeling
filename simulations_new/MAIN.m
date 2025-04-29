%% Initialize MRST
clear; close all; clc
mrstModule add incomp spe10 
linsolve = @mldivide;
gravity on
set(0,'DefaultFigureWindowStyle','docked')
cwd = pwd; cd('D:\MATLAB\mrst-2024a\'); startup; cd(cwd); clear cwd

realization = 114;

%% Grid generation
G  = cartGrid([128 128 1], [1000 1000 10]*meter);
G  = computeGeometry(G);

%% Rock and Fluid properties
dataset = load('por_perm_facies_1000x128x128.mat');
poro = reshape(dataset.poro_norm(realization,:,:), [128*128,1]);
perm = 10.^reshape(dataset.perm_norm(realization,:,:), [128*128,1]) *milli*darcy;
channel = reshape(dataset.facies_norm(realization,:,:), [128*128,1]);

rock = makeRock(G, perm, poro);

fluid = initSimpleFluid('mu',  [1, 5]*centi*poise, ...
                        'rho', [1000, 850]*kilogram/meter^3, ...
                        'n'  , [2, 2]);

%% Wells
I = 64;
J = 64;
R = 500*stb/day;    
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

I = [1 128 1   128];
J = [1 1   128 128];
p_prod = 1000;
P = [p_prod p_prod p_prod p_prod]*psia;
nPW = (1:numel(I))+max(nIW);
for i = 1 : numel(I)
    W = verticalWell(W, G, rock, I(i), J(i), [], ...
        'Type', 'bhp', ...
        'InnerProduct', 'ip_tpf', ...
        'Val', P(i), ...
        'Radius', 0.1, ...
        'name', ['P', int2str(i)], ...
        'Comp_i', [0, 1]);
end

%% Visualization Por/Perm
figure(1)
plotCellData(G, rock.poro, 'EdgeAlpha', 0.2);
plotWell(G,W,'color','black')
title('Porosity [v/v]'); colorbar; colormap('jet'); %view(-10,85)

figure(2)
plotCellData(G, log10(convertTo(rock.perm,milli*darcy)), 'EdgeAlpha', 0.2);
plotWell(G,W,'color','black')
title('log_{10} Permeability [mD]'); colorbar; colormap('jet'); %view(-10,85)

figure(3)
plotCellData(G, channel); colormap('jet')
plotWell(G,W,'color','black')
title('Fluvial Channels'); colorbar; %view(-10,85)

%% Transmissibilities and initial state
trans = computeTrans(G, rock);
rSol  = initState(G, W, 3000*psia, [0.1,0.9]);
rSol = incompTPFA(rSol, G, trans, fluid, 'wells', W, 'LinSolve', linsolve);

%% Main loop
T = 10*year;     %total time
dT = 1*year/4;   %time step
total_timesteps = T/dT;
dTplot = dT;

% Prepare plotting of saturations
figure(4)
movegui(figure(4));
h=colorbar; colormap jet; %view(-10,85)
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
    figure(4)
    delete([hs_p, ha_p])
    hs_p = plotCellData(G, rSol.s(:,2), 'EdgeAlpha', 0.2);
    plotWell(G, W, 'height', 10, 'color', 'k'); %view(-10,85)
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

%% Plot rates
figure(5)

subplot(221); 
plot(convertTo(Prod.t, year), convertTo(Prod.opr(:,2:5), stb/day))
title('OPR'); xlabel('time [year]'); ylabel('stb/day')

subplot(222); 
plot(convertTo(Prod.t, year), convertTo(Prod.wpr(:,2:5), stb/day))
title('WPR'); xlabel('time [year]'); ylabel('stb/day')

subplot(223);
plot(convertTo(Prod.t, year), Prod.wc(:,2:5))
title('Water Cut'); xlabel('time [year]'); ylim([0,1])

subplot(224);
plotCellData(G, log10(convertTo(rock.perm,milli*darcy)), 'EdgeAlpha', 0.2);
plotWell(G,W,'color','black'); colormap('jet'); %view(-10,85)

%% END