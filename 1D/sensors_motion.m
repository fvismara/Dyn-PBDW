function [pos_vec] = sensors_motion(V,x_sens0,xgrid,sigma,move_sensors)
    
    % Given the reduced basis V and the initial position of the sensors 
    % x_sens0, determine the position of the sensors at each time. pos_vec 
    % is a m-by-(Ntse+1) matrix, where m is the number of sensors and Ntse 
    % is the number of time steps at which state estimation is performed
    
    Ntse = size(V,3) - 1;
    number_of_measurements = numel(x_sens0);
    
    if move_sensors
        pos_vec = zeros(number_of_measurements,Ntse+1);
        x_sens = x_sens0;
        for i = 1:Ntse+1
            % Select basis functions at timestep i
            vi = V(:,:,i);
            % Run gradient descent to find sensor positions
            x_sens = grad_descent(vi,x_sens,xgrid,sigma);
            % Store locations of the sensors
            pos_vec(:,i) = x_sens;
        end
    else
        % If the sensors are not moved, their positions coincide with the
        % initial position at all time steps
        pos_vec = repmat(x_sens0,1,Ntse+1);
    end
    
end

