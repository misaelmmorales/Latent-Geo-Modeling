%% Run SNESIM from mGSTAT for random fluvial channel realizations
% This script is really slow for a large number of realizations. Run once.
% Specify:
% nx, ny = number of cells in x- and y-directions
% nsim = number of realizations to simulate

% user-defined constants
dims = [128 128 16];
num_sim = 2;

% run snesim
S = snesim_init;
S.pdf_target = [0.700 0.300];
S.nx   = dims(1);
S.ny   = dims(2);
S.nz   = dims(3);
S.rseed = 1234123;
S.nmulgrids = 5;
S.tau1 = 1;
S.tau2 = 1; 
S.nsim = num_sim;

S = snesim(S);

% Collect and save results
channel_3d = reshape(S.D, [], num_sim);
csvwrite('channel_3d.csv', channel_3d);

%% Temp
clear;clc

S=snesim_init;

S.nx=128;
S.ny=128;
S.nsim = 10;
S.rseed=1;
S=snesim(S);
