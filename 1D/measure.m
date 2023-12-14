function [Bq,BqD,Bp,BpD] = measure(v,x,xgrid,sigma)
    
    % Compute the matrices Bq,BqD,Bp,Bpd given the basis functions v and
    % the sensors locations x
            
    % In the case of local averages, the goal is to compute
    % (\omega_j,v)_{L^2}, where v is either vq, vp or their
    % derivatives. By interpolating the nodal values on piecewise
    % constant functions, everything boils down to computing
    % (\omega_j,\phi_i)_{L^2} for all p.w. constant functions \phi_i,
    % i=1,\dots,N. Since we know the expressions of \omega_j and
    % \phi_i, these integrals can be computed by hand. Then the
    % resulting matrix is multiplied by the expansion coefficients,
    % i.e. the nodal values, contained in the vector v.

    Lx = xgrid(end);
    dx = xgrid(2) - xgrid(1);
    N = numel(xgrid);

    vq = v(1:end/2,:);
    vp = v(end/2+1:end,:);

    xkphalf = -Lx + ((1:N) + 1/2)*dx;
    xkmhalf = -Lx + ((1:N) - 1/2)*dx;
    a = (xkmhalf - x(:))/(sigma*sqrt(2));
    b = (xkphalf - x(:))/(sigma*sqrt(2));
    Mv = (erf(b) - erf(a))/(2*sqrt(dx));
    Bq = Mv*vq;
    Bp = Mv*vp;

    Mvx = -(exp(-b.^2)-exp(-a.^2))/(sqrt(dx*2*pi*sigma^2));
    BqD = Mvx*vq;
    BpD = Mvx*vp;
        
end

