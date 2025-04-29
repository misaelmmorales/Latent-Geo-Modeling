%% Initialize MRST
clear; close all; clc
mrstModule add incomp spe10 ad-core ad-blackoil ad-props mrst-gui
gravity on
set(0,'DefaultFigureWindowStyle','docked')
cwd = pwd; cd('D:\MATLAB\mrst-2024a\'); startup; cd(cwd); clear cwd

realization = 589;

%% Run Simulations
nr=1000; 
nx=128; ny=128; nz=1;
Dx=800; Dy=800; Dz=8;

G  = cartGrid([nx ny nz], [Dx Dy Dz]*meter);
G  = computeGeometry(G);

dataset = load('por_perm_facies_1000x128x128.mat');
poro_all = reshape(dataset.poro_norm, [nr,nx*ny]);
perm_all = reshape(dataset.perm_norm, [nr,nx*ny]);
channel = reshape(dataset.facies_norm, [nr,nx*ny]);

fluid = initSimpleFluid('mu',  [1, 5]*centi*poise, ...
                        'rho', [1000, 850]*kilogram/meter^3, ...
                        'n'  , [2, 2]);

kk = reshape(10.^perm_all(realization,:)*milli*darcy, [nx*ny,1]);
pp = reshape(poro_all(realization,:), [nx*ny,1]);
rock = makeRock(G, kk, pp);
clear kk pp

W = [];
W = verticalWell(W, G, rock, 64, 64, [], ...
    'Type', 'rate', ...
    'InnerProduct', 'ip_tpf', ...
    'Val', 1000*stb/day, ...
    'Comp_i', [1, 0], ...
    'name', 'I1');

I = [1 128 1   128];
J = [1 1   128 128];
for z = 1 : numel(I)
    W = verticalWell(W, G, rock, I(z), J(z), [], ...
        'Type', 'bhp', ...
        'InnerProduct', 'ip_tpf', ...
        'Val', 500*psia, ...
        'name', ['P', int2str(z)], ...
        'Comp_i', [0, 1]);
end

%% Simulation
T = 10*year;
dT = year/4;

state0 = initResSol(G, 3000*psia, [0.2, 0.8]);
model = TwoPhaseOilWaterModel(G, rock, fluid);

schedule = simpleSchedule(rampupTimesteps(T, dT, 0), 'W', W);
[ws, states] = simulateScheduleAD(state0, model, schedule);

%% Save results
save(sprintf('results/states/states_%d.mat', realization), 'states');
save(sprintf('results/wellsol/wellsol_%d.mat', realization), 'ws');

%% Plot
figure(1); clf
plotToolbar(G, states, 'edgecolor', 'k', 'edgealpha',0.5);
colormap jet; colorbar; plotWell(G,W); view(35,85)