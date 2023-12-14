% Choose test case. 'SCH':Schrodinger, 'SWE':Shallow Water Equation
test_problem = 'SWE';

%%%%%%%%%%%%%% Load full order solution and reduced basis %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Nothing should be modified here %%%%%%%%%%%%%%%%%%%%%%
switch test_problem
    case 'SCH'
        % Physical parameters. Spatial domain [-Lx,Lx], time domain [0,T]
        Lx = 20*pi;
        T = 20;
        % Load basis and reference solution. Reduced basis constructed 
        % using 10 training parameters per direction. Reference solution 
        % computed using 9 test parameters per direction. Full and reduced 
        % systems are solved using Nt=20000 time steps, but we perform 
        % state estimation every 10 time steps so that we only consider 
        % Ntse=2000 time steps. This is mainly to save memory, since saving
        % all time steps for all 81 test parameters would result in large 
        % data matrices. Results obtained with Ntse=Nt are qualitatively 
        % identical
        load('C:\Users\20205114\Desktop\dynamical-state-estimation\1D Schrodinger\Data\snaps_reduced.mat')
        load('C:\Users\20205114\Desktop\dynamical-state-estimation\1D Schrodinger\Data\snaps_full.mat')
        Avec = Avec(:,:,1:10:end);
    case 'SWE'
        % Physical parameters. Spatial domain [-Lx,Lx], time domain [0,T]
        Lx = 30;
        T = 10;
        % Load basis and reference solution. Reduced basis constructed 
        % using 5 training parameters per direction. Reference solution 
        % computed using 4 test parameters per direction. Full and reduced 
        % systems are solved using Nt=12000 time steps. Here we take again
        % Ntse=Nt/10=1200
        load('C:\Users\20205114\Desktop\dynamical-state-estimation\1D Shallow Water\Data\snaps_reduced.mat')
        load('C:\Users\20205114\Desktop\dynamical-state-estimation\1D Shallow Water\Data\snaps_full.mat')
        Avec = Avec(:,:,1:10:end);
        snaps_full_test = snaps_full_test(:,:,1:10:end);
end
% Vector of test parameters corresponding to the columns of snaps_full_test
dse.test_params = test_params;
% Dimension of the full order problem 2N
N = size(snaps_full_test,1)/2;
dse.N = N;
% Dimension of the reduced problem 2n
n = size(Avec,2)/2;
% Number of test parameters
dse.ntest = size(snaps_full_test,2);
% Number of time steps at which state estimation is performed. As mentioned
% before, here Ntse=Nt/10
dse.Ntse = size(snaps_full_test,3) - 1;
% Time interval between each reconstruction
dt = T/dse.Ntse;
% Grid spacing
dx = 2*Lx/N;
% Space grid
dse.xgrid = -Lx+(dx:dx:2*Lx);
% Hamiltonian
switch test_problem
    case 'SCH'
        DXX = (1/dx^2)*spdiags([ones(N,1),-2*ones(N,1),ones(N,1)],[-1,0,1],N,N);
        DXX(1,end) = 1/dx^2;
        DXX(end,1) = 1/dx^2;
        dse.Ham = @(Y,ee) (dx/2)*(-diag(Y(1:end/2,:)'*DXX*Y(1:end/2,:) + ...
            Y(end/2+1:end,:)'*DXX*Y(end/2+1:end,:))'/2 - ...
            (ee/4)*sum((Y(1:end/2,:).^2+Y(end/2+1:end,:).^2).^2));
    case 'SWE'
        % Hamiltonian
        DX = (1/(2*dx))*spdiags([ones(N,1),-ones(N,1)],[1,-1],N,N);
        DX(1,end) = -1/(2*dx);
        DX(end,1) = 1/(2*dx);
        dse.Ham = @(Y,ee) (dx/2)*sum(Y(1:end/2,:).*((DX*Y(end/2+1:end,:)).^2) + ...
            Y(1:end/2,:).^2);
end        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% State estimation
dse.number_of_measurements = 10;
% Width of local average sigma
dse.sigma = .1;
% Initial position of the sensors. 'e':equispaced,'r':random,'n':grid 
% points where the first basis function is largest
dse.initial_placement = 'e';
% Select interval to place sensors at the initial time when
% initial_placement is either 'e' or 'r'. This interval is
% [-Lx+2*s1*Lx,-Lx+2*s2*Lx]. Therefore we require 0<s1<s2<1. The initial
% placement should not be symmetric around the origin to avoid bad
% conditioning. In the numerical experiments we set s1=0.481, s2=0.521 for
% Schrodinger, s1=0.441, s2=0.561 for SWE
dse.s1 = 0.441;
dse.s2 = 0.561;
% If true, dynamic placement of the sensors. If false, do not move them
dse.move_sensors = true;

% Run state estimation
dse = dynamical_state_estimation(dse,snaps_full_test,Avec);
%% Plot x coordinate of sensors over time and position of main bump, beta and errors
% Select test parameter to plot position of the bump.
iparam = 7; % SWE:7 SCH:41

figure
subplot(1,3,1)
% Plot x coordinates for each sensor
for i = 1:dse.number_of_measurements
    plot(linspace(0,T,dse.Ntse+1),dse.sens_loc(i,:))
    hold on
end

% Select snapshots corresponding to the chosen test parameter
snaps_full = squeeze(snaps_full_test(:,iparam,:));
% Compute location of the maximum at each time and plot it
switch test_problem
    case 'SCH'
        % In the Schrodinger case there is only one main bump
        [~,posmax] = max(sqrt(snaps_full(1:end/2,:).^2+snaps_full(end/2+1:end,:).^2));
        plot(linspace(0,T,dse.Ntse+1),-Lx+posmax*dx,'r--','LineWidth',2)
    case 'SWE'
        % In the shallow water case there are two bumps moving in opposite
        % directions so we get the coordinates of both
        posmax = zeros(2,dse.Ntse+1);
        for i = 1:dse.Ntse+1
            [~,posmaxi] = sort(snaps_full(1:end/2,i),'descend');
            posmax(:,i) = posmaxi(1:2);
        end
        posmax = sort(posmax);
        plot(linspace(0,T,dse.Ntse+1),-Lx+posmax(1,:)*dx,'r--','LineWidth',1.5)
        plot(linspace(0,T,dse.Ntse+1),-Lx+posmax(2,:)*dx,'r--','LineWidth',1.5)
end

% Plot beta
subplot(1,3,2)
semilogy(linspace(0,T,dse.Ntse+1),dse.beta_vec,'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
title('$$\beta(t)$$','interpreter','latex')
ylim([1e-06,1e+00])

% Compute errors
proj_err = zeros(dse.ntest,dse.Ntse+1);
se_err = zeros(dse.ntest,dse.Ntse+1);
upbound_err = zeros(dse.ntest,dse.Ntse+1);
for iparam = 1:dse.ntest
    y = squeeze(snaps_full_test(:,iparam,:));
    ys = squeeze(dse.ys_vec(:,iparam,:));
    [proj_erri,se_erri] = compute_errors(y,Avec,ys,dse.xgrid);
    upbound_erri = proj_erri./dse.beta_vec;
    proj_err(iparam,:) = proj_erri;
    se_err(iparam,:) = se_erri;
    upbound_err(iparam,:) = upbound_erri;
    iparam
end
Ese = max(se_err,[],1);
Eu = max(upbound_err,[],1);
El = max(proj_err,[],1);

% Plot 3 errors
subplot(1,3,3)
semilogy(linspace(0,T,dse.Ntse+1),Ese,'LineWidth',2)
hold on
semilogy(linspace(0,T,dse.Ntse+1),Eu,'LineWidth',2)
semilogy(linspace(0,T,dse.Ntse+1),El,'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
title('$$\mathcal{E}(t)$$','interpreter','latex')
legend('$$||y-y^*||$$','$$\frac{||y-AA^\top y||}{\beta}$$','$$||y-AA^\top y||$$','interpreter','latex','location','SouthEast')
ylim([1e-04,1e+04])

%% Plot exact and reconstructed solution at time tplot
% Choose test parameter to plot and time
iparam = 13; % SWE:13 SCH:81
tplot = 7;   % SWE:7 SCH:20=T

itplot = floor(tplot/dt) + 1;
figure
y = squeeze(snaps_full_test(:,iparam,:));
ystar = squeeze(dse.ys_vec(:,iparam,:));
qt = y(1:end/2,itplot); 
pt = y(end/2+1:end,itplot) + tplot*(strcmp(test_problem,'SWE'));
qst = ystar(1:end/2,itplot); 
pst = ystar(end/2+1:end,itplot) + tplot*(strcmp(test_problem,'SWE'));
subplot(1,2,1)
plot(dse.xgrid,qt)
hold on
plot(dse.xgrid,qst)
legend('$$q(x,T)$$','$$q^*(x,T)$$','interpreter','latex')
subplot(1,2,2)
plot(dse.xgrid,pt)
hold on
plot(dse.xgrid,pst)
legend('$$p(x,T)$$','$$p^*(x,T)$$','interpreter','latex')

%% Plot Hamiltonian conservation
% Choose test parameter to plot
iparam = 5; % SWE:5 SCH:28

Hyi = dse.Hy_vec(iparam,:);
Hysi = dse.Hys_vec(iparam,:);

% This first plot shows the evolution of the Hamiltonian in the full and
% reconstructed solution for the chosen test parameter iparam
figure
subplot(1,3,1)
plot(linspace(0,T,dse.Ntse+1),Hysi,'LineWidth',2)
hold on
plot(linspace(0,T,dse.Ntse+1),Hyi,'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
legend('$$H(y^*(t))$$','$$H(y(t))$$','interpreter','latex')

% In the same plot we show the error in the Hamiltonian between full and
% reconstructed solution (left) and the variation of the Hamiltonian with
% respect to the initial condition in the full and reconstructed solution
% (right)
subplot(1,3,2)
eH = max(abs(dse.Hy_vec-dse.Hys_vec));
semilogy(linspace(0,T,dse.Ntse+1),eH,'LineWidth',2)
legend('$$|H(u)-H(u^*)|$$','interpreter','latex')
subplot(1,3,3)
dH = max(abs(dse.Hy_vec-dse.Hy_vec(:,1)));
dHs = max(abs(dse.Hys_vec-dse.Hys_vec(:,1)));
semilogy(linspace(0,T,dse.Ntse+1),dH,'LineWidth',2)
hold on
semilogy(linspace(0,T,dse.Ntse+1),dHs,'LineWidth',2)
legend('$$|H(u)-H(u_0)|$$','$$|H(u^*)-H(u^*_0)|$$','interpreter','latex')

%% Plot exact solution and sensors over time (movie)
% Choose parameter to plot
iparam = 16;

% Plot full order solution corresponding to the chosen test parameter and
% motion of the sensors.
snaps_full = squeeze(snaps_full_test(:,iparam,:));
qmax = max(max(snaps_full(1:end/2,:)));
qmin = min(min(snaps_full(1:end/2,:)));
pmax = max(max(snaps_full(end/2+1:end,:)));
pmin = min(min(snaps_full(end/2+1:end,:))) + T*strcmp(test_problem,'SWE');
figure
for i = 1:dse.Ntse+1
    Yi = snaps_full(:,i);
    sens_loci = dse.sens_loc(:,i);
    [qmeasn,~,pmeasn,~] = measure(Yi,sens_loci,dse.xgrid,dse.sigma);
    
    subplot(1,2,1)
    plot(dse.xgrid,Yi(1:end/2))
    hold on
    plot(sens_loci,sqrt(dx)*qmeasn,'kx','MarkerSize',10)
    xlim([-Lx,Lx])
    ylim([qmin,qmax])
    hold off
    title(sprintf('t=%.4f s',(i-1)*dt),'interpreter','latex')
    
    subplot(1,2,2)
    plot(dse.xgrid,Yi(end/2+1:end)+(i-1)*dt*strcmp(test_problem,'SWE'))
    hold on
    plot(sens_loci,sqrt(dx)*pmeasn+(i-1)*dt*strcmp(test_problem,'SWE'),'kx','MarkerSize',10)
    xlim([-Lx,Lx])
    ylim([pmin,pmax])
    hold off
    title(sprintf('t=%.4f s',(i-1)*dt),'interpreter','latex')
    
    pause(0.001)
end

%% Plot full order solution at 4 times
% Choose test parameter to plot
iparam = 7; % SCH:41, SWE:7
% Choose index of 4 time steps at which the solution should be plotted
iplot = [1,301,601,1001]; % SCH:[1,501,1501,2001], SWE:[1,301,601,1001]

tiledlayout(2,4,'Padding','none','TileSpacing','compact'); 
for i = 1:4    
    nexttile    
    plot(dse.xgrid,snaps_full_test(1:end/2,iparam,iplot(i)),'LineWidth',1.5)
    title(sprintf('t=%.2f',(iplot(i)-1)*dt),'interpreter','latex')
    xlim([-Lx,Lx])
    switch test_problem
        case 'SWE'
            ylim([0.98,1.14])
        case 'SCH'
            ylim([-0.2,1.5])
    end
end
for i = 1:4    
    nexttile    
    plot(dse.xgrid,snaps_full_test(end/2+1:end,iparam,iplot(i))+(iplot(i)-1)*dt*strcmp(test_problem,'SWE'),'LineWidth',1.5)
    xlim([-Lx,Lx])
    switch test_problem
        case 'SWE'
            ylim([-0.15,0.02])
        case 'SCH'
            ylim([-0.5,0.5])
    end
end