%%%%%%%%%%%%%% Load full order solution and reduced basis %%%%%%%%%%%%%%%%%
% Physical parameters. Spatial domain [-Lx,Lx]x[-Ly,Ly], time domain [0,T]
Lx = 8;
Ly = 8;
T = 10;
% Load basis and reference solution. Reduced basis constructed 
% using 5 training parameters per direction. Reference solution 
% computed using 4 test parameters per direction. Full and reduced 
% systems are solved using Nt=10000 time steps, but we perform 
% state estimation every 10 time steps so that we only consider 
% Ntse=1000 time steps.
load('C:\Users\20205114\Desktop\dynamical-state-estimation\2D Shallow Water\Data\snaps_reduced.mat')
load('C:\Users\20205114\Desktop\dynamical-state-estimation\2D Shallow Water\Data\snaps_full.mat')
snaps_full_test = snaps_full_test(:,:,1:10:end);
Avec = Avec(:,:,1:10:end);
% Vector of test parameters corresponding to the columns of snaps_full_test
dse.test_params = test_params;
% Dimension of the full order problem 2N
N = size(snaps_full_test,1)/2;
Nx = 50; dse.Nx = Nx;
Ny = 50; dse.Ny = Ny;
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
dx = 2*Lx/Nx;
dy = 2*Ly/Ny;
% Space grid
dse.xgrid = -Lx+(dx:dx:2*Lx);
dse.ygrid = -Ly+(dy:dy:2*Ly);
% Hamiltonian
DX1d = diag( ones(Nx-1,1),1) - diag( ones(Nx-1,1),-1);
DX1d(1,end) = -1;
DX1d(end,1) = 1;
DX1d = DX1d/(2*dx);
DX = zeros(Nx*Ny,Nx*Ny);
for i = 1:Ny
    DX((i-1)*Nx + 1: i*Nx,(i-1)*Nx + 1: i*Nx) = DX1d;
end
DX = sparse(DX);

DY = zeros(Nx*Ny,Nx*Ny);
for i = 1:Ny
    if i < Ny
        DY((i-1)*Nx + 1 : i*Nx,i*Nx + 1 : (i+1)*Nx) = eye(Nx);
    end
    if i > 1
        DY((i-1)*Nx + 1 : i*Nx, (i-2)*Nx + 1 : (i-1)*Nx) = -eye(Nx);
    end
end
DY(1:Nx,end-Nx+1:end) = -eye(Nx);
DY(end-Nx+1:end,1:Nx) = eye(Nx);
DY = sparse(DY/(2*dy));

dse.Ham = @(Y) (dx/2)*sum(Y(1:end/2,:).*((DX*Y(end/2+1:end,:)).^2+(DY*Y(end/2+1:end,:)).^2) + Y(1:end/2,:).^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% State estimation
dse.number_of_measurements = 10;
% Width of local average
dse.sigma = 0.1;
% Select random placements in [ax1,ax2]x[by1,by2], where ax1=-Lx+s1x*2*Lx
% and ax2=-Lx+s2x*2*Lx (and similarly for by1 and by2). Therefore
% 0<=s1x<s2x<=1 and 0<=s1y<s2y<=1
dse.s1x = 0.45;
dse.s2x = 0.55;
dse.s1y = 0.45;
dse.s2y = 0.55;
dse.move_sensors = true;
%%%%%%%%%%%%%%%%%%%%%%%% END OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% profile on
dse = dynamical_state_estimation(dse,snaps_full_test,Avec);
% profile viewer
%% Plot sensors and exact solution at 4 time instants
iparam = 8;
tplot = [1,301,601,1001];

figure
tiledlayout(2,4,'Padding','none','TileSpacing','compact'); 
ind = 1;
snaps_full = squeeze(snaps_full_test(:,iparam,:));
for i = tplot
    nexttile
    surf(dse.xgrid,dse.ygrid,reshape(snaps_full(1:end/2,i),Nx,Ny)')
    shading interp
    view(2)
    colorbar
    hold on
    x_sens = dse.sens_loc(1:end/2,i);
    y_sens = dse.sens_loc(end/2+1:end,i);
    plot3(x_sens,y_sens,10*ones(numel(x_sens),1),'rx','LineWidth',2)
%     title(sprintf('t=%.4f s',(i-1)*dt),'interpreter','latex')
    xlim([-Lx,Lx])
    ylim([-Ly,Ly])
    ind = ind+1;
end

for i = tplot
    nexttile
    surf(dse.xgrid,dse.ygrid,reshape(snaps_full(end/2+1:end,i)+(i-1)*dt,Nx,Ny)')
    shading interp
    view(2)
    colorbar
    hold on
    x_sens = dse.sens_loc(1:end/2,i);
    y_sens = dse.sens_loc(end/2+1:end,i);
    plot3(x_sens,y_sens,10*ones(numel(x_sens),1),'rx','LineWidth',2)
%     title(sprintf('t=%.4f s',(i-1)*dt),'interpreter','latex')
    xlim([-Lx,Lx])
    ylim([-Ly,Ly])
    ind = ind+1;
end

%% Plot reconstruction error and error bound
figure
proj_err = zeros(dse.ntest,dse.Ntse+1);
se_err = zeros(dse.ntest,dse.Ntse+1);
upbound_err = zeros(dse.ntest,dse.Ntse+1);
for iparam = 1:dse.ntest
    snaps_full = squeeze(snaps_full_test(:,iparam,:));
    ystar = squeeze(dse.ys_vec(:,iparam,:));
    [proj_erri,se_erri] = compute_errors(snaps_full,Avec,ystar,dse.xgrid,dse.ygrid);
    proj_err(iparam,:) = proj_erri;
    se_err(iparam,:) = se_erri;
    upbound_err(iparam,:) = proj_erri./dse.beta_vec;
    iparam
end
Ese = max(se_err,[],1);
El = max(proj_err,[],1);
Eu = max(upbound_err,[],1);

subplot(1,2,1)
semilogy(linspace(0,T,dse.Ntse+1),dse.beta_vec,'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
legend('$$\beta(t)$$','interpreter','latex')
xlim([0,T])
subplot(1,2,2)
semilogy(linspace(0,T,dse.Ntse+1),Ese,'LineWidth',2)
hold on
semilogy(linspace(0,T,dse.Ntse+1),Eu,'LineWidth',2)
semilogy(linspace(0,T,dse.Ntse+1),El,'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
legend('$$\overline{\mathcal{E}}_{se}$$','$$\overline{\mathcal{E}}_u$$','$$\overline{\mathcal{E}}_l$$','interpreter','latex')
xlim([0,T])

valmat = [linspace(0,T,dse.Ntse+1); El; Eu; Ese; dse.beta_vec];
%% Plot Hamiltonian
% Choose test parameter to plot
figure
iparam = 8;

Hyi = dse.Hy_vec(iparam,:);
Hysi = dse.Hys_vec(iparam,:);

% This first plot shows the conservation of the Hamiltonian in the full and
% reconstructed solution
subplot(1,3,1)
plot(linspace(0,T,dse.Ntse+1),Hysi,'LineWidth',2)
hold on
plot(linspace(0,T,dse.Ntse+1),Hyi,'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
legend('$$H(y^*(t))$$','$$H(y(t))$$','interpreter','latex')

% In the same plot we show the error in the Hamiltonian between full and
% reconstructed solution (center) and the variation of the Hamiltonian with
% respect to the initial condition in the full and reconstructed solution
% (right)
subplot(1,3,2)
eHs = max(abs(dse.Hy_vec-dse.Hys_vec));
semilogy(linspace(0,T,dse.Ntse+1),eHs,'LineWidth',2)
legend('$$|H(u)-H(u^*)|$$','interpreter','latex')
subplot(1,3,3)
dH = max(abs(dse.Hy_vec-dse.Hy_vec(:,1)));
dHs = max(abs(dse.Hys_vec-dse.Hys_vec(:,1)));
semilogy(linspace(0,T,dse.Ntse+1),dH,'LineWidth',2)
hold on
semilogy(linspace(0,T,dse.Ntse+1),dHs,'LineWidth',2)
legend('$$|H(u)-H(u_0)|$$','$$|H(u^*)-H(u^*_0)|$$','interpreter','latex')

% This second plot shows the contributions to the Hamiltonian conservation
% error in the reconstructed solution |H(y*)-H(y0*)|. The blue line is
% bounded by the sum of the other three lines. In particular the error
% denoted by the orange line is bounded by the Lipschitz constant of H
% times the reconstruction error
figure
semilogy(linspace(0,T,dse.Ntse+1),abs(Hysi-Hysi(1)),'LineWidth',2)
hold on
semilogy(linspace(0,T,dse.Ntse+1),abs(Hysi-Hyi),'LineWidth',2)
semilogy(linspace(0,T,dse.Ntse+1),abs(Hyi-Hyi(1)),'LineWidth',2)
semilogy([0,T],[abs(Hyi(1)-Hysi(1)),abs(Hyi(1)-Hysi(1))],'LineWidth',2)
xlabel('$$t$$','interpreter','latex')
legend('$$|H(y^*)-H(y_0^*)|$$','$$|H(y^*)-H(y)|$$','$$|H(y)-H(y_0)|$$','$$|H(y_0)-H(y_0^*)|$$','interpreter','latex','location','SouthEast')

valmat = [linspace(0,T,dse.Ntse+1); Hyi; Hysi; eHs; dH; dHs];
%% Plot sensors and exact solution over time
iparam = 1;

figure
snaps_full = squeeze(snaps_full_test(:,iparam,:));
for i = 1:dse.Ntse+1
    surf(dse.xgrid,dse.ygrid,reshape(snaps_full(1:end/2,i),Nx,Ny)')
    shading interp
    view(2)
    hold on
    x_sens = dse.sens_loc(1:end/2,i);
    y_sens = dse.sens_loc(end/2+1:end,i);
    plot3(x_sens,y_sens,10*ones(numel(x_sens),1),'rx','LineWidth',2)
    title(sprintf('t=%.4f s',(i-1)*dt),'interpreter','latex')
    pause(0.001)
    hold off
end  

%% Plot sensors trajectories
figure
for i = 1:dse.number_of_measurements
    xi_loc = dse.sens_loc(i,:);
    yi_loc = dse.sens_loc(i+dse.number_of_measurements,:);
    indtime_perx = find(abs(xi_loc(2:end)-xi_loc(1:end-1))>Lx);
    indtime_pery = find(abs(yi_loc(2:end)-yi_loc(1:end-1))>Ly);
    indtime_per = [0,unique([indtime_perx,indtime_pery]),dse.Ntse];
    for j = 2:numel(indtime_per)
        curr_int = indtime_per(j-1)+1:indtime_per(j);
        plot(xi_loc(curr_int),yi_loc(curr_int))
        hold on
    end
    hold on
    plot(xi_loc(1),yi_loc(1),'rx','LineWidth',1.5)
    plot(xi_loc(401),yi_loc(401),'rs','LineWidth',1.5)
    plot(xi_loc(801),yi_loc(801),'r^','LineWidth',1.5)
end
% [~,maxvec] = max(sqrt(snaps_full(1:end/2,:).^2+snaps_full(end/2+1:end,:).^2));
% plot(mod(maxvec,Nx)*dx-Lx,mod(maxvec,Ny)*dy-Ly,'r--','LineWidth',1.5)
xlim([-Lx,Lx])
ylim([-Ly,Ly])

%% Plot exact solution, reconstruction and projection
iparam = 1;
tplot = dse.Ntse+1;

figure
subplot(2,3,1)
surf(dse.xgrid,dse.ygrid,reshape(snaps_full_test(1:end/2,iparam,end),Nx,Ny)')
shading interp
% view(2)
xlim([-Lx,Lx])
ylim([-Ly,Ly])
subplot(2,3,2)
surf(dse.xgrid,dse.ygrid,reshape(dse.ys_vec(1:end/2,iparam,end),Nx,Ny)')
shading interp
% view(2)
xlim([-Lx,Lx])
ylim([-Ly,Ly])
subplot(2,3,3)
surf(dse.xgrid,dse.ygrid,reshape(Avec(1:end/2,:,end)*(Avec(:,:,end)'*snaps_full_test(:,iparam,end)),Nx,Ny)')
shading interp
% view(2)
xlim([-Lx,Lx])
ylim([-Ly,Ly])
subplot(2,3,4)
surf(dse.xgrid,dse.ygrid,reshape(snaps_full_test(end/2+1:end,iparam,end),Nx,Ny)')
shading interp
% view(2)
xlim([-Lx,Lx])
ylim([-Ly,Ly])
subplot(2,3,5)
surf(dse.xgrid,dse.ygrid,reshape(dse.ys_vec(end/2+1:end,iparam,end),Nx,Ny)')
shading interp
% view(2)
xlim([-Lx,Lx])
ylim([-Ly,Ly])
subplot(2,3,6)
surf(dse.xgrid,dse.ygrid,reshape(Avec(end/2+1:end,:,end)*(Avec(:,:,end)'*snaps_full_test(:,iparam,end)),Nx,Ny)')
shading interp
% view(2)
xlim([-Lx,Lx])
ylim([-Ly,Ly])

%% save .gif
iparam = 1;

fig = figure;
idx = 1;
snaps_full = squeeze(snaps_full_test(:,iparam,:));
for i = 1:reconstruct_state_every:Nt+1
    surf(xgrid,ygrid,reshape(sqrt(snaps_full(1:end/2,i).^2+snaps_full(end/2+1:end,i).^2),Nx,Ny)')
    shading interp
    view(2)
    hold on
    x_sens = pos_vec(1:end/2,idx);
    y_sens = pos_vec(end/2+1:end,idx);
    plot3(x_sens,y_sens,10*ones(numel(x_sens),1),'rx','LineWidth',2)
    title("t="+(i-1)*dt+"s",'interpreter','latex')
    pause(0.001)
    drawnow
    frame = getframe(fig);
    im{idx} = frame2im(frame);
    hold off
    idx = idx+1;
end
filename = "SWE_sigma0p01_v5.gif"; % Specify the output file name
for idx = 1:size(pos_vec,2)
    [AA,map] = rgb2ind(im{idx},256);
    if idx == 1
        imwrite(AA,map,filename,"gif","LoopCount",Inf,"DelayTime",.1);
    else
        imwrite(AA,map,filename,"gif","WriteMode","append","DelayTime",.1);
    end
end
