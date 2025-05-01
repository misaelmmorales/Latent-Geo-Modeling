%% Initialize MRST
clear; close all; clc
cwd = pwd; cd('D:\MATLAB\mrst-2024a\'); startup; cd(cwd); clear cwd
mrstModule add incomp spe10 ad-core ad-blackoil ad-props mrst-gui
gravity on
set(0,'DefaultFigureWindowStyle','docked')

%% Global parameters
nr=1000; 
nx=128; ny=128; nz=1;
Dx=800; Dy=800; Dz=8;

G  = cartGrid([nx ny nz], [Dx Dy Dz]*meter);
G  = computeGeometry(G);

fluid = initSimpleFluid('mu',  [1, 5]*centi*poise, ...
                        'rho', [1000, 850]*kilogram/meter^3, ...
                        'n'  , [2, 2]);

dataset = load('por_perm_facies_1000x128x128.mat');
poro_all = reshape(dataset.poro_norm, [nr,nx*ny]);
perm_all = reshape(dataset.perm_norm, [nr,nx*ny]);
channel = reshape(dataset.facies_norm, [nr,nx*ny]);
rock_all.poro = poro_all;
rock_all.perm = perm_all;
rock_all.channel = channel;

%% Run main loop

parfor i=1:1000
    [states] = run_simulation(i, G, fluid, rock_all);
end