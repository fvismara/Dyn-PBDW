function [ystar,beta] = reconstruct_solution(y,V,xgrid,sigma,x_sens)
    
    % Reconstruct solution using basis functions V and sensor locations x_sens 
    
    x_sens = x_sens(:);
    Lx = xgrid(end);
    
    % Measure the full solution and basis functions at the measurement points
    [Bq,~,Bp,~] = measure(V,x_sens,xgrid,sigma);
    [yq,~,yp,~] = measure(y,x_sens,xgrid,sigma);
    
    % Compute Aq=Ap
    s = (x_sens+x_sens')/2;
    d = (x_sens-x_sens')/2;
    sa = (-Lx-s)/sigma;
    sb = (Lx-s)/sigma;
    Aq = ((erf(sb)-erf(sa))/(4*sqrt(pi)*sigma)).*exp(-(d/sigma).^2);
    
    % Solve least square problem B'*A^(-1)*B*z=B'*A^(-1)y, where B=[Bq;Bp],
    % A=[Aq,0;0,Aq] and y=[yq;yp]. We perform QR decomposition of
    % sqrt(A^(-1))*B
    [U,S,~] = svd(Aq);
    sqrtA_times_B = [U*(sqrt(S)\(U'*Bq));U*(sqrt(S)\(U'*Bp))];
    [Q,R] = qr(sqrtA_times_B,0);
    beta = min(svd(sqrtA_times_B));
    w = [U*(sqrt(S)\(U'*yq));U*(sqrt(S)\(U'*yp))];
    zstar = R\(Q'*w);
    
    % Reconstruct solution (only the component in V2n)
    ystar = V*zstar;
    
end

