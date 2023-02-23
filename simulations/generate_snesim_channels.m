%% Run SNESIM from mGSTAT for random fluvial channel realizations
% This script is really slow for a large number of realizations. Run once.
% Specify:
% nx, ny = number of cells in x- and y-directions
% nsim = number of realizations to simulate

% user-defined constants
dims = [128 128 1];
num_sim = 1000;

% run snesim
S = snesim_init;
S.nx   = dims(1);
S.ny   = dims(2);
S.rseed = 1234123;
S.nmulgrids = 5;
S.tau1 = 1;
S.tau2 = 1; 
S.nsim = num_sim;

S = snesim(S);

% Collect and save results
channel_all = reshape(S.D, [], num_sim);
csvwrite('channel_all.csv', channel_all);
