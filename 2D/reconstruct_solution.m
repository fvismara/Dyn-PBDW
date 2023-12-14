function [ystar,beta] = reconstruct_solution(y,V,xgrid,ygrid,sigma,x_sens,y_sens)

    % Reconstruct solution using basis functions V and sensor locations x_sens
    
    x_sens = x_sens(:);
    y_sens = y_sens(:);
    Lx = xgrid(end);
    Ly = ygrid(end);
    
    % Measure basis functions and solution
    [Gq,~,~,Gp,~,~] = measure(V,x_sens,y_sens,xgrid,ygrid,sigma);
    [yq,~,~,yp,~,~] = measure(y,x_sens,y_sens,xgrid,ygrid,sigma);
    
    % Compute Aq=Ap=G(W,W)
    sx = (x_sens+x_sens')/2; sy = (y_sens+y_sens')/2;
    dx = (x_sens-x_sens')/2; dy = (y_sens-y_sens')/2;
    sax = (-Lx-sx)/sigma; say = (-Ly-sy)/sigma;
    sbx = (Lx-sx)/sigma; sby = (Ly-sy)/sigma;
    Ax1d = (1/(4*sqrt(pi)*sigma))*(erf(sbx)-erf(sax)).*exp(-(dx/sigma).^2);
    Ay1d = (1/(4*sqrt(pi)*sigma))*(erf(sby)-erf(say)).*exp(-(dy/sigma).^2);
    A = Ax1d.*Ay1d;
    
    % Solve least square problem B'*A^(-1)*B*z=B'*A^(-1)y, where B=[Bq;Bp],
    % A=[GWW,0;0,GWW] and y=[yq;yp]. In order to do this, we perform QR 
    % decomposition of M=sqrt(A^(-1))*B
    [U,S,~] = svd(A);
    M = [U*(sqrt(S)\(U'*Gq));U*(sqrt(S)\(U'*Gp))];
    [Q,R] = qr(M);
    beta = min(svd(M));
    w = [U*(sqrt(S)\(U'*yq));U*(sqrt(S)\(U'*yp))];
    zstar = R\(Q'*w);
    
    % Component of the reconstructed solution in V2n
    ystar = V*zstar;
    
end

