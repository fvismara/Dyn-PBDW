function [xnew,ynew,A] = grad_descent(V,x,y,xgrid,ygrid,sigma)
    
    Lx = xgrid(end);
    Ly = ygrid(end);
    
    x = x(:);
    y = y(:);

    % Initial condition for gradient descent are the old positions
    xk = x;
    yk = y;
    posk = [xk;yk];

    % Measure basis functions to build matrices GWV, GWVx and GWVy
    [Bq,BqxD,BqyD,Bp,BpxD,BpyD] = measure(V,xk,yk,xgrid,ygrid,sigma);
    
    % Assemble Aq=Ap=A, AqxD=ApxD=AxD and AqyD=ApyD=AyD
    sx = (xk+xk')/2; sy = (yk+yk')/2;
    dx = (xk-xk')/2; dy = (yk-yk')/2;
    sax = (-Lx-sx)/sigma; say = (-Ly-sy)/sigma;
    sbx = (Lx-sx)/sigma; sby = (Ly-sy)/sigma;
    Ax1d = (1/(4*sqrt(pi)*sigma))*(erf(sbx)-erf(sax)).*exp(-(dx/sigma).^2);
    Ay1d = (1/(4*sqrt(pi)*sigma))*(erf(sby)-erf(say)).*exp(-(dy/sigma).^2);
    A = Ax1d.*Ay1d;
    AxD1d = -(1/(4*pi*sigma^3))*(-dx*sqrt(pi).*(erf(sbx)-erf(sax))-sigma*(exp(-sbx.^2)-exp(-sax.^2))).*exp(-(dx/sigma).^2);
    AyD1d = -(1/(4*pi*sigma^3))*(-dy*sqrt(pi).*(erf(sby)-erf(say))-sigma*(exp(-sby.^2)-exp(-say.^2))).*exp(-(dy/sigma).^2);
    AxD = AxD1d.*Ay1d;
    AyD = Ax1d.*AyD1d;
    
    % u is the singular vector of M corresponding to the smallest s.v.
    [~,S,VV] = svd(Bq'*(A\Bq)+Bp'*(A\Bp),'econ');
    u = VV(:,end);
    % beta^2 is the eigenvalue corresponding to u (since M is spd,
    % sing.values = eigenvalues
    betasquared = S(end,end);
    
    % Compute gradient of beta wrt positions    
    Ainv_Bq_u = A\(Bq*u);
    Ainv_Bp_u = A\(Bp*u);
    grad = [2*Ainv_Bq_u.*(BqxD*u - AxD*Ainv_Bq_u) + ...
        2*Ainv_Bp_u.*(BpxD*u - AxD*Ainv_Bp_u); ...
        2*Ainv_Bq_u.*(BqyD*u - AyD*Ainv_Bq_u) + ...
        2*Ainv_Bp_u.*(BpyD*u - AyD*Ainv_Bp_u)];

    % Number of iterations
    nit = 0;
    delta_betasquared_rel = 1;

    % While the relative variation of beta^2 is larger than tolerance and
    % number of iteration is less than maximum, do gradient descent
    % iterations
    while delta_betasquared_rel > 10^(-8) && nit < 10

        betasquared_old = betasquared;
        pos_old = posk;
        % Move in the direction of the gradient using learning rate 10000
        alpha = 10000;
        posk = pos_old + alpha*grad/norm(grad);
        % Get x and y coordinates in the rectangle
        xk = mod(posk(1:end/2)+Lx,2*Lx)-Lx;
        yk = mod(posk(end/2+1:end)+Ly,2*Ly)-Ly;

        % Compute B,Bx,By,A corresponding to the new positions
        [Bq,BqxD,BqyD,Bp,BpxD,BpyD] = measure(V,xk,yk,xgrid,ygrid,sigma);
        sx = (xk+xk')/2; sy = (yk+yk')/2;
        dx = (xk-xk')/2; dy = (yk-yk')/2;
        sax = (-Lx-sx)/sigma; say = (-Ly-sy)/sigma;
        sbx = (Lx-sx)/sigma; sby = (Ly-sy)/sigma;
        Ax1d = (1/(4*sqrt(pi)*sigma))*(erf(sbx)-erf(sax)).*exp(-(dx/sigma).^2);
        Ay1d = (1/(4*sqrt(pi)*sigma))*(erf(sby)-erf(say)).*exp(-(dy/sigma).^2);
        A = Ax1d.*Ay1d;
        
        % Recompute u corresponding to the new M
        [~,S,VV] = svd(Bq'*(A\Bq)+Bp'*(A\Bp),'econ');
        betasquared = S(end,end);
        
        % Backtracking line search: while the Armijo condition is not
        % satisfied ...
        while betasquared < betasquared_old + alpha*norm(grad)/2 && alpha > 10^(-10)
            % Reduce learning rate
            alpha = alpha/2;
            % Recompute positions
            posk = pos_old + alpha*grad/norm(grad);
            xk = mod(posk(1:end/2)+Lx,2*Lx)-Lx;
            yk = mod(posk(end/2+1:end)+Ly,2*Ly)-Ly;
            % Recompute matrices
            [Bq,BqxD,BqyD,Bp,BpxD,BpyD] = measure(V,xk,yk,xgrid,ygrid,sigma);
            sx = (xk+xk')/2; sy = (yk+yk')/2;
            dx = (xk-xk')/2; dy = (yk-yk')/2;
            sax = (-Lx-sx)/sigma; say = (-Ly-sy)/sigma;
            sbx = (Lx-sx)/sigma; sby = (Ly-sy)/sigma;
            Ax1d = (1/(4*sqrt(pi)*sigma))*(erf(sbx)-erf(sax)).*exp(-(dx/sigma).^2);
            Ay1d = (1/(4*sqrt(pi)*sigma))*(erf(sby)-erf(say)).*exp(-(dy/sigma).^2);
            A = Ax1d.*Ay1d;
            % Recompute beta
            [~,S,VV] = svd(Bq'*(A\Bq)+Bp'*(A\Bp),'econ');
            betasquared = S(end,end);
        end
        
        % Build matrices AxD and AyD
        AxD1d = -(1/(4*pi*sigma^3))*(-dx*sqrt(pi).*(erf(sbx)-erf(sax))-sigma*(exp(-sbx.^2)-exp(-sax.^2))).*exp(-(dx/sigma).^2);
        AyD1d = -(1/(4*pi*sigma^3))*(-dy*sqrt(pi).*(erf(sby)-erf(say))-sigma*(exp(-sby.^2)-exp(-say.^2))).*exp(-(dy/sigma).^2);
        AxD = AxD1d.*Ay1d;
        AyD = Ax1d.*AyD1d;

        % u is the singular vector of M corresponding to the smallest s.v.
        u = VV(:,end);
        % Compute gradient of beta wrt positions
        Ainv_Bq_u = A\(Bq*u);
        Ainv_Bp_u = A\(Bp*u);
        grad = [2*Ainv_Bq_u.*(BqxD*u - AxD*Ainv_Bq_u) + ...
            2*Ainv_Bp_u.*(BpxD*u - AxD*Ainv_Bp_u); ...
            2*Ainv_Bq_u.*(BqyD*u - AyD*Ainv_Bq_u) + ...
            2*Ainv_Bp_u.*(BpyD*u - AyD*Ainv_Bp_u)];
        % Increment number of iterations
        nit = nit + 1;
        
        delta_betasquared_rel = norm(betasquared_old-betasquared)/norm(betasquared_old);

    end

    % Set the new positions of the sensors equal to the last iteration
    xnew = mod(xk+Lx,2*Lx)-Lx;
    ynew = mod(yk+Ly,2*Ly)-Ly;
    
end

