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

fluid = initSimpleADIFluid('phases',    'WO', ...
                           'mu',        [1,    5]*centi*poise, ...
                           'rho',       [1000, 850], ...
                           'n',         [2,    2], ...
                           'c',         [0,    1e-5]/barsa, ...
                           'cR',        4e-10/barsa);

%% Timesteps and injection rate [T/# because # injectors]
total_time = 10*year;
irate = sum(poreVolume(G, rock)) / (total_time*5);

%% Wells
makeInj = @(W, name, I, J) verticalWell(W, G, rock, I, J, [],...
    'Name', name, 'radius', 5*inch, 'sign', 1, 'InnerProduct', 'ip_tpf', ...
    'Type', 'rate', 'Val', irate, 'comp_i', [1, 0]);

makeProd = @(W, name, I, J) verticalWell(W, G, rock, I, J, [],...
    'Name', name, 'radius', 5*inch, 'sign', -1, 'InnerProduct', 'ip_tpf', ...
    'Type', 'bhp', 'Val', 1000*psia, 'comp_i', [1, 1]/2);

W = [];
W = makeInj(W, 'I1', 1, 1);
W = makeInj(W, 'I2', 1, 48);
W = makeInj(W, 'I3', 24, 24);
W = makeInj(W, 'I4', 48, 1);
W = makeInj(W, 'I5', 48, 48);
W = makeProd(W, 'P1', 1, 24);
W = makeProd(W, 'P2', 24, 1);
W = makeProd(W, 'P3', 24, 48);
W = makeProd(W, 'P4', 48, 24);

%% Visualize
figure
plotCellData(G, rock.poro, 'EdgeAlpha', 0); plotWell(G,W)
view(-25,70); colormap jet; cb=colorbar; cb.Label.String='Porosity';
title(['Realization ', num2str(realization)])
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]')

figure
plotCellData(G, log10(convertTo(rock.perm(:,1),milli*darcy)), 'EdgeAlpha', 0) 
plotWell(G,W)
view(-25,70); colormap jet; cb=colorbar; cb.Label.String='Log-Permeability';
title(['Realization ', num2str(realization)])
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]')

%% Schedule
% Set up a schedule with two different controls. In the first control, the
% injectors are set to the copy of the wells we made earlier where water
% was injected. In the second control, gas is injected instead.
dT_target = (4/12)*year;
dt = rampupTimesteps(total_time, dT_target, 10);
schedule = simpleSchedule(dt, 'W', W);

%% Model
model = TwoPhaseOilWaterModel(G, rock, fluid);
state0 = initResSol(G, 2000*psia, [0.10, 0.90]);
[wellSol, states] = simulateScheduleAD(state0, model, schedule);

%% Collect Results
% timestamps_yr = convertTo(cumsum(dt), year); %run just once
% save data/timestamp_yr timestamps_yr         %run just once

% [bhp, opr, wpr, wct] = deal(zeros(40,9));
% [pres, satu]         = deal(zeros(40,G.cells.num));
% for i=1:length(dt)
%     for j=1:numel(W)
%         bhp(i,j) = convertTo(wellSol{i}(j).bhp, psia);
%         opr(i,j) = abs(convertTo(wellSol{i}(j).qOs, stb/day));
%         wpr(i,j) = abs(convertTo(wellSol{i}(j).qWs, stb/day));
%         wct(i,j) = wellSol{i}(j).wcut;
%     end
%     pres(i,:) = convertTo(states{i}.pressure, psia);
%     satu(i,:) = states{i}.s(:,2);
% end
% production = cat(3, bhp, opr, wpr, wct);
% 
% prod_name = sprintf('data/production/production_%d.mat', realization);
% poro_name = sprintf('data/porosity/porosity_%d.mat', realization);
% perm_name = sprintf('data/permeability/permeability_%d.mat', realization);
% pres_name = sprintf('data/pressure/pressure_%d.mat', realization);
% satu_name = sprintf('data/saturation/saturation_%d.mat', realization);
% 
% save(fullfile(pwd(), prod_name), 'production')
% save(fullfile(pwd(), poro_name), 'porosity')
% save(fullfile(pwd(), perm_name), 'perm_md')
% save(fullfile(pwd(), pres_name), 'pres')
% save(fullfile(pwd(), satu_name), 'satu')

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