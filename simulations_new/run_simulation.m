function [states] = run_simulation(i, G, fluid, rock_all)
    nx = G.cartDims(1);
    ny = G.cartDims(2);

    % make rock
    perm_all = rock_all.perm;
    poro_all = rock_all.poro;
    kk = reshape(10.^perm_all(i,:)*milli*darcy, [nx*ny,1]);
    pp = reshape(poro_all(i,:), [nx*ny,1]);
    rock = makeRock(G, kk, pp);

    % make wells
    W = [];
    W = verticalWell(W, G, rock, 64, 64, [], ...
        'Type', 'rate', ...
        'InnerProduct', 'ip_tpf', ...
        'Val', 1000*stb/day, ...
        'Comp_i', [1, 0], ...
        'name', 'I1');
    I = [1 128 1   128];
    J = [1 1   128 128];
    for z=1:numel(I)
        W = verticalWell(W, G, rock, I(z), J(z), [], ...
            'Type', 'bhp', ...
            'InnerProduct', 'ip_tpf', ...
            'Val', 500*psia, ...
            'name', ['P', int2str(z)], ...
            'Comp_i', [0, 1]);
    end

    % run simulation
    T  = 10*year;
    dT = year/4;
    state0       = initResSol(G, 3000*psia, [0.2, 0.8]);
    model        = TwoPhaseOilWaterModel(G, rock, fluid);
    schedule     = simpleSchedule(rampupTimesteps(T, dT, 0), 'W', W);
    [ws, states] = simulateScheduleAD(state0, model, schedule);

    save(sprintf('results/states/states_%d.mat', i), 'states');
    save(sprintf('results/wellsol/wellsol_%d.mat', i), 'ws');
    
end

