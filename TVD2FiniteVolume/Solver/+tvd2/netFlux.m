function net=netFlux(faceFlux)
%NETFLUX Integral net influx for each cell.
% Positive FACEFLUX points from left to right.

validateattributes(faceFlux,{'numeric'},{'2d','real','finite'});
if size(faceFlux,2)<2
    error('tvd2:FaceFlux','At least two face fluxes are required.');
end
net=faceFlux(:,1:end-1)-faceFlux(:,2:end);
end

