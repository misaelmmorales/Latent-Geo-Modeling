function [G, rock, fluid, Prod, pres, satu] = run_simulations(realization)
%RUN_SIMULATIONS Summary of this function goes here
%   Detailed explanation goes here rSol.s(:,2)

linsolve = @mldivide;
gravity on

G  = cartGrid([128 128 1], [1000 1000 10]*meter);
G  = computeGeometry(G);
perm_all = readmatrix('perm_all.csv');
channel_all = readmatrix('channel_all.csv');
channel = reshape(reshape(channel_all(:,realization), [128,128])', [], 1);
channel(channel==0) = 0.8;
poro = channel.*10.^((perm_all(:,realization)-7)/10);
perm = channel.*(10.^perm_all(:,realization))*milli*darcy;
permeability = convertTo(perm, milli*darcy); %for export
rock = makeRock(G, perm, poro);
fluid = initSimpleFluid('mu',  [1, 5]*centi*poise, ...
                        'rho', [1000, 850]*kilogram/meter^3, ...
                        'n'  , [2, 2]);
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

trans = computeTrans(G, rock);
rSol  = initState(G, W, 3000*psia, [0.2,0.8]);
rSol = incompTPFA(rSol, G, trans, fluid, 'wells', W, 'LinSolve', linsolve);

T = 15*year;     %total time
dT = 1*year/3;   %time step
total_timesteps = T/dT;
dTplot = dT;

t  = 0;  plotNo = 1;
wres = cell([1, 4]);

Prod = struct('t'  , []                  , ...
              'vpt', zeros([0, numel(W)]), ...
              'opr', zeros([0, numel(W)]), ...
              'wpr', zeros([0, numel(W)]), ...
              'wc' , zeros([0, numel(W)]));
satu = zeros(G.cartDims(1)*G.cartDims(2), total_timesteps);
pres = zeros(G.cartDims(1)*G.cartDims(2), total_timesteps);

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
    [wres{:}] = prodCurves(W, rSol, fluid);
    Prod      = append_wres(Prod, t, wres{:});
    satu(:, plotNo) = rSol.s(:,2);
    pres(:, plotNo) = convertTo(rSol.pressure, psia);
    plotNo = plotNo+1;
end

prod_name = sprintf("response_production/production_%d.mat", realization);
satu_name = sprintf("response_saturation/saturation_%d.mat", realization);
pres_name = sprintf("response_pressure/pressure_%d.mat", realization);
poro_name = sprintf("features_porosity/porosity_%d.mat", realization);
perm_name = sprintf("features_permeability/permeability_%d.mat", realization);

save(fullfile(pwd(), prod_name), 'Prod')
save(fullfile(pwd(), satu_name), 'satu')
save(fullfile(pwd(), pres_name), 'pres')
save(fullfile(pwd(), poro_name), 'poro')
save(fullfile(pwd(), perm_name), 'permeability')

end

