function [dse] = dynamical_state_estimation(dse,snaps_full_test,Avec)

    % Compute evolution of sensors and reconstruct solution for all test
    % parameters
    
    % Retrieve parameters
    Ntse = dse.Ntse;
    ntest = dse.ntest;
    Nx = dse.Nx;
    Ny = dse.Ny;
    N = Nx*Ny;
    xgrid = dse.xgrid;
    ygrid = dse.ygrid;
    Lx = xgrid(end);
    Ly = ygrid(end);
    s1x = dse.s1x;
    s2x = dse.s2x;
    s1y = dse.s1y;
    s2y = dse.s2y;
    number_of_measurements = dse.number_of_measurements;
    move_sensors = dse.move_sensors;
    Ham = dse.Ham;
    sigma = dse.sigma;
    
    % Initialize vectors to store quantities
    beta_vec = zeros(1,Ntse+1);
    Hy_vec = zeros(ntest,Ntse+1);
    Hys_vec = zeros(ntest,Ntse+1);
    ys_vec = zeros(2*N,ntest,Ntse+1);

    % Define initial placement of the sensors: random in the rectangle
    rng(5)
    x_sens0 = -Lx + 2*s1x*Lx + 2*Lx*(s2x-s1x)*rand(number_of_measurements,1);
    y_sens0 = -Ly + 2*s1y*Ly + 2*Ly*(s2y-s1y)*rand(number_of_measurements,1);

    % Loop over test parameters
    for iparam = 1:ntest
        
        % The evolution of the sensors does not depend on the test
        % parameter, so we only compute it for the first one
        if iparam == 1
            sens_loc = sensors_motion(Avec,x_sens0(:),y_sens0(:),xgrid,ygrid,sigma,move_sensors);
        end

        for i = 1:Ntse+1
            
            % Select full solution, reduced basis and sensors positions at
            % the current time
            y = snaps_full_test(:,iparam,i);
            A = Avec(:,:,i);
            x_sensi = sens_loc(1:end/2,i);
            y_sensi = sens_loc(end/2+1:end,i);

            % Reconstruct solution and compute beta
            [ystar,beta] = reconstruct_solution(y,A,xgrid,ygrid,sigma,x_sensi,y_sensi);

            % Store beta and reconstructed solution
            beta_vec(i) = beta;
            ys_vec(:,iparam,i) = ystar;

        end
        
        % Evaluate Hamiltonian at the full order solution 
        Hy_vec(iparam,:) = Ham(snaps_full_test(:,iparam,:));
        % Evaluate Hamiltonian at the reconstructed solution
        Hys_vec(iparam,:) = Ham(ys_vec(:,iparam,:));
        
        iparam
        
    end
    
    dse.beta_vec = beta_vec;
    dse.ys_vec = ys_vec;
    dse.Hy_vec = Hy_vec;
    dse.Hys_vec = Hys_vec;
    dse.sens_loc = sens_loc;

end

