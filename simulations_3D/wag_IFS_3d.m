%% MRST
clear;clc;close all
mrstModule add ad-core ad-blackoil ad-props spe10 mrst-gui

%% Grid, Rock and Fluid
G=cartGrid([48,48,8], [1000,1000,80]*meter); G=computeGeometry(G);

load facies_maps_48_48_8.mat
load logperm_48_48_8.mat
clear Label
realization = 41;

channel  = reshape(TI(realization,:),[],1);
porosity = channel .* 10.^((perm(:,realization)-7)/10);
permx    = channel .* 10.^perm(:,realization) * milli * darcy;
perm_md  = convertTo(permx, milli*darcy); %for export
permeability(:,1) = permx;
permeability(:,2) = permx;
permeability(:,3) = permx*0.1;

rock.poro = porosity;
rock.perm = permeability;

fluid = initSimpleADIFluid('phases',    'WOG', ...
                           'mu',        [1,    5,    0.15]*centi*poise, ...
                           'rho',       [1000, 850,  250], ...
                           'n',         [2,    2,    2], ...
                           'c',         [0,    1e-5, 1e-3]/barsa, ...
                           'cR',        4e-10/barsa);

%% Timesteps and injection rate [T/# because # injectors]
total_time = 10*year;
irate = sum(poreVolume(G, rock)) / (total_time*5);

%% Wells
makeInj = @(W, name, I, J, compi) verticalWell(W, G, rock, I, J, [],...
    'Name', name, 'radius', 5*inch, 'sign', 1, 'InnerProduct', 'ip_tpf', ...
    'Type', 'rate', 'Val', irate, 'comp_i', compi);

makeProd = @(W, name, I, J) verticalWell(W, G, rock, I, J, [],...
    'Name', name, 'radius', 5*inch, 'sign', -1, 'InnerProduct', 'ip_tpf', ...
    'Type', 'bhp', 'Val', 1000*psia, 'comp_i', [1, 1, 1]/3);

W = [];
W = makeInj(W, 'I1', 1, 1, []);
W = makeInj(W, 'I2', 1, 48, []);
W = makeInj(W, 'I3', 24, 24, []);
W = makeInj(W, 'I4', 48, 1, []);
W = makeInj(W, 'I5', 48, 48, []);
W = makeProd(W, 'P1', 1, 24);
W = makeProd(W, 'P2', 24, 1);
W = makeProd(W, 'P3', 24, 48);
W = makeProd(W, 'P4', 48, 24);

% Create two copies of wells: 1 = water injection and 2 = gas injection
[W_water, W_gas] = deal(W);
for i = 1:numel(W)
    if W(i).sign < 0 
        continue   %Skip producers
    end
    W_water(i).compi = [1, 0, 0];
    W_gas(i).compi   = [0, 0, 1];
end

%% Visualize
figure
plotCellData(G, rock.poro); plotWell(G,W)
view(-25,70); colormap jet; cb=colorbar; cb.Label.String='Porosity';
title(['Realization ', num2str(realization)])
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]')

figure
plotCellData(G, log10(convertTo(rock.perm(:,1),milli*darcy))); plotWell(G,W)
view(-25,70); colormap jet; cb=colorbar; cb.Label.String='Log-Permeability';
title(['Realization ', num2str(realization)])
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]')

%% Schedule
% Set up a schedule with two different controls. In the first control, the
% injectors are set to the copy of the wells we made earlier where water
% was injected. In the second control, gas is injected instead.

dT_target = (4/12)*year;
dt = rampupTimesteps(total_time, dT_target, 10);

schedule = struct();
schedule.control = [struct('W', W_water); struct('W', W_gas)];
schedule.step.val = dt;
schedule.step.control = (mod(cumsum(dt), 2*dT_target) >= dT_target) + 1;

%% Model
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', false, 'vapoil', false);
state0 = initResSol(G, 2000*psia, [0.10, 0.90, 0]);
[wellSol, states] = simulateScheduleAD(state0, model, schedule);

%% Post-sim Viewer
figure;
plotToolbar(G, states); plotWell(G, W);
axis tight; view(-25,70); colormap jet; title('States Solution')

figure;
plotWellSol(wellSol, dt)

%{
figure, clf
ctrl = repmat(schedule.step.control', 2, 1);
x = repmat(cumsum(dt/year)', 2, 1);
y = repmat([0; 1], 1, size(ctrl, 2));
surf(x, y, ctrl)
colormap(jet)
view(0, 90)
axis equal tight
set(gca, 'YTick', []);
xlabel('Time [year]')
title('Control changes over time: Red for gas injection, blue for water')
%}

%% END