function [proj_err,se_err] = compute_errors(snaps_full,Avec,ystar_vec,xgrid)
    
    % Compute projection error and state estimation error for one test
    % parameter, assuming that the reconstruction belongs to V2n and the 
    % columns of Avec are an orthonormal basis of V2n. The norms
    % in V are approximated by interpolating the nodal values of the full 
    % order solution and of the reduced basis functions using piecewise 
    % constant functions on each grid interval
    
    Ntse = size(snaps_full,2) - 1;
    dx = xgrid(2) - xgrid(1);

    proj_err = zeros(1,Ntse+1);
    se_err = zeros(1,Ntse+1);
    for i = 1:Ntse+1
        Ai = Avec(:,:,i);
        yi = snaps_full(:,i);
        ysi = ystar_vec(:,i);
        
        y_minus_Py = yi - Ai*(Ai'*yi);
        y_minus_ystar = yi - ysi;
        
        proj_err(i) = sqrt(dx)*sqrt(y_minus_Py'*y_minus_Py);
        se_err(i) = sqrt(dx)*sqrt(y_minus_ystar'*y_minus_ystar);
    end
    
end