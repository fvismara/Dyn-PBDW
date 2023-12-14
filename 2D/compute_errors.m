function [proj_err,se_err] = compute_errors(y,V,ys,xgrid,ygrid)
    
    % Given the "exact" solution y, the reconstruction ys and the reduced
    % basis V, computes the state estimation error, and the projection
    % error in the L2 norm. The integral is approximated by interpolating
    % the nodal values using piecewise constant functions
    
    Ntse = size(y,2) - 1;
    
    dx = xgrid(2) - xgrid(1);
    dy = ygrid(2) - ygrid(1);

    proj_err = zeros(1,Ntse+1);
    se_err = zeros(1,Ntse+1);
    for i = 1:Ntse+1
        yi = y(:,i);
        ysi = ys(:,i);
        Vi = V(:,:,i);
        
        y_minus_Py = yi - Vi*(Vi'*yi);
        y_minus_ystar = yi - ysi;
        
        proj_err(i) = sqrt(dx*dy)*sqrt(y_minus_Py'*y_minus_Py);
        se_err(i) = sqrt(dx*dy)*sqrt(y_minus_ystar'*y_minus_ystar);
    end
    
end