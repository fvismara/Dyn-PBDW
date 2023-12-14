function [dse] = dynamical_state_estimation(dse,snaps_full_test,Avec)

    % Compute evolution of sensors and reconstruct solution for all test
    % parameters
    
    % Retrieve parameters
    Ntse = dse.Ntse;
    ntest = dse.ntest;
    N = dse.N;
    xgrid = dse.xgrid;
    Lx = xgrid(end);
    s1 = dse.s1;
    s2 = dse.s2;
    number_of_measurements = dse.number_of_measurements;
    move_sensors = dse.move_sensors;
    Ham = dse.Ham;
    initial_placement = dse.initial_placement;
    sigma = dse.sigma;
    test_params = dse.test_params;
    
    % Initialize vectors to store quantities
    beta_vec = zeros(1,Ntse+1);
    Hy_vec = zeros(ntest,Ntse+1);
    Hys_vec = zeros(ntest,Ntse+1);
    ys_vec = zeros(2*N,ntest,Ntse+1);

    % Define initial placement of the sensors
    switch initial_placement
        case 'r'
            % Random
            x_sens0 = -Lx+2*s1*Lx+2*(s2-s1)*Lx*rand(1,number_of_measurements);
        case 'e'
            % Equispaced
            x_sens0 = linspace(-Lx+2*s1*Lx,-Lx+2*s2*Lx,number_of_measurements);
    end

    % Compute evolution of the sensors using x_sens0 as initial condition
    sens_loc = sensors_motion(Avec,x_sens0(:),xgrid,sigma,move_sensors);
    
    % Perform state estimation on all test parameters
    for iparam = 1:ntest

        for i = 1:Ntse+1

            y = snaps_full_test(:,iparam,i);
            A = Avec(:,:,i);
            sens_loci = sens_loc(:,i);

            % Reconstruct solution and compute beta
            [ystar,beta] = reconstruct_solution(y,A,xgrid,sigma,sens_loci);

            % Store beta and reconstructed solution
            beta_vec(i) = beta;
            ys_vec(:,iparam,i) = ystar;

        end
        
        % Evaluate Hamiltonian at the full order solution 
        Hy_vec(iparam,:) = Ham(snaps_full_test(:,iparam,:),test_params(2,iparam));
        % Evaluate Hamiltonian at the reconstructed solution
        Hys_vec(iparam,:) = Ham(ys_vec(:,iparam,:),test_params(2,iparam));
        
        iparam
        
    end
    
    % Save quantities
    dse.beta_vec = beta_vec;
    dse.ys_vec = ys_vec;
    dse.Hy_vec = Hy_vec;
    dse.Hys_vec = Hys_vec;
    dse.sens_loc = sens_loc;

end

