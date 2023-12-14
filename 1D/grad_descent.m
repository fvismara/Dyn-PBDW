function [xnew] = grad_descent(V,x,xgrid,sigma)
    
    % Given the old positions x and the reduced basis functions V at a
    % given time, run gradient descent to find the optimal sensor locations
    % at the given time
    
    Lx = xgrid(end);

    % Initialize positions x^(0)
    xk = x(:);
    posk = xk;
    
    % Build matrices G(W,V) and G(W,Vx) at x^(k)
    [Bq,BqD,Bp,BpD] = measure(V,xk,xgrid,sigma);
    % Build A and AD. Their expressions can be computed by hand
    s = (xk+xk')/2;
    d = (xk-xk')/2;
    sa = (-Lx-s)/sigma;
    sb = (Lx-s)/sigma;
    A = (1/(4*sqrt(pi)*sigma))*(erf(sb)-erf(sa)).*exp(-(d/sigma).^2);
    AD = (1/(4*pi*sigma^3))*(-sqrt(pi)*(erf(sb)-erf(sa)).*d+sigma*(exp(-sa.^2)-exp(-sb.^2))).*exp(-(d/sigma).^2);

    % u is the singular vector of M corresponding to the smallest s.v.    
    [~,S,VV] = svd(Bq'*(A\Bq)+Bp'*(A\Bp),'econ');
    u = VV(:,end);
    betasquared = S(end,end);

    % Compute gradient of beta wrt x. Here Aq=Ap=A and AqD=ApD=AD
    Ainv_Bq_u = A\(Bq*u);
    Ainv_Bp_u = A\(Bp*u);
    grad = 2*Ainv_Bq_u.*(BqD*u - AD*Ainv_Bq_u) + ...
        + 2*Ainv_Bp_u.*(BpD*u - AD*Ainv_Bp_u);

    % Initialize number of iterations and relative variation of beta^2
    nit = 0;
    delta_betasquared_rel = 1; 

    while delta_betasquared_rel > 10^(-8) && nit < 10

        betasquared_old = betasquared;
        pos_old = posk;
        
        % Move in the direction of the gradient with step size alpha
        alpha = 100;
        posk = pos_old + alpha*grad/norm(grad);
        % Bring sensors back in [-Lx,Lx] in case they left the domain
        xk = mod(posk+Lx,2*Lx)-Lx;
        
        % Measure basis functions at the current locations xk
        [Bq,BqD,Bp,BpD] = measure(V,xk,xgrid,sigma);
        % Compute A at the current locations xk
        s = (xk+xk')/2;
        d = (xk-xk')/2;
        sa = (-Lx-s)/sigma;
        sb = (Lx-s)/sigma;
        A = (1/(4*sqrt(pi)*sigma))*(erf(sb)-erf(sa)).*exp(-(d/sigma).^2);
        % Compute beta^2 corresponding to the current locations xk
        [~,S,VV] = svd(Bq'*(A\Bq)+Bp'*(A\Bp),'econ');
        betasquared = S(end,end);
        
        % Backtracking line search: while the Armijo condition is not
        % satisfied, divide alpha by 2 and repeat the process
        while betasquared < betasquared_old + alpha*norm(grad)/2 && alpha > 10^(-10)
            alpha = alpha/2;
            posk = pos_old + alpha*grad/norm(grad);
            xk = mod(posk+Lx,2*Lx)-Lx;
            [Bq,BqD,Bp,BpD] = measure(V,xk,xgrid,sigma);
            s = (xk+xk')/2;
            d = (xk-xk')/2;
            sa = (-Lx-s)/sigma;
            sb = (Lx-s)/sigma;
            A = (1/(4*sqrt(pi)*sigma))*(erf(sb)-erf(sa)).*exp(-(d/sigma).^2);
            [~,S,VV] = svd(Bq'*(A\Bq)+Bp'*(A\Bp),'econ');
            betasquared = S(end,end);
        end

        % Build matrix AqD=ApD=AD at the locations xk
        AD = (1/(4*pi*sigma^3))*(-sqrt(pi)*(erf(sb)-erf(sa)).*d+sigma*(exp(-sa.^2)-exp(-sb.^2))).*exp(-(d/sigma).^2);
        % u is the singular vector of M corresponding to the smallest s.v.
        u = VV(:,end);
        % Compute gradient of beta wrt x
        Ainv_Bq_u = A\(Bq*u);
        Ainv_Bp_u = A\(Bp*u);
        grad = 2*Ainv_Bq_u.*(BqD*u - AD*Ainv_Bq_u) + ...
            + 2*Ainv_Bp_u.*(BpD*u - AD*Ainv_Bp_u);
        % Increment number of iterations
        nit = nit + 1;
        
        % Update relative variation of beta^2
        delta_betasquared_rel = norm(betasquared_old-betasquared)/norm(betasquared_old);

    end

    % Set the new positions of the sensors equal to the last iteration, and
    % move them back inside the domain in case they left it
    xnew = mod(xk+Lx,2*Lx)-Lx;
    
end



