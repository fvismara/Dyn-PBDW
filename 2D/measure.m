function [qmeasn,qxmeasn,qymeasn,pmeasn,pxmeasn,pymeasn] = measure(v,x,y,xgrid,ygrid,sigma)

    % Measure function(s) v at the points of coordinates (x,y)
    Nx = numel(xgrid); Ny = numel(ygrid);
    dx = xgrid(2) - xgrid(1); dy = ygrid(2) - ygrid(1);

    Lx = xgrid(end); Ly = ygrid(end);

    vq = v(1:end/2,:);
    vp = v(end/2+1:end,:);

    xkphalf = -Lx + ((1:Nx) + 1/2)*dx;
    xkmhalf = -Lx + ((1:Nx) - 1/2)*dx;
    ykphalf = -Ly + ((1:Ny) + 1/2)*dy;
    ykmhalf = -Ly + ((1:Ny) - 1/2)*dy;
    ax = (xkmhalf - x(:))/(sigma*sqrt(2));
    bx = (xkphalf - x(:))/(sigma*sqrt(2));
    Mvx = (erf(bx) - erf(ax))/(2*sqrt(dx));
    Mvdx = -(exp(-bx.^2)-exp(-ax.^2))/(sqrt(dx*2*pi*sigma^2));
    ay = (ykmhalf - y(:))/(sigma*sqrt(2));
    by = (ykphalf - y(:))/(sigma*sqrt(2));
    Mvy = (erf(by) - erf(ay))/(2*sqrt(dy));
    Mvdy = -(exp(-by.^2)-exp(-ay.^2))/(sqrt(dy*2*pi*sigma^2));
    
    Mvx = sparse(Mvx.*(abs(Mvx)>10^(-12)));
    Mvy = sparse(Mvy.*(abs(Mvy)>10^(-12)));
    Mvdx = sparse(Mvdx.*(abs(Mvdx)>10^(-12)));
    Mvdy = sparse(Mvdy.*(abs(Mvdy)>10^(-12)));

    mWV = repmat(Mvx,1,Ny).*repelem(Mvy,1,Nx);
    mWVx = repmat(Mvdx,1,Ny).*repelem(Mvy,1,Nx);
    mWVy = repmat(Mvx,1,Ny).*repelem(Mvdy,1,Nx);
    qmeasn = mWV*vq;
    qxmeasn = mWVx*vq;
    qymeasn = mWVy*vq;
    pmeasn = mWV*vp;
    pxmeasn = mWVx*vp;
    pymeasn = mWVy*vp;
    
end

