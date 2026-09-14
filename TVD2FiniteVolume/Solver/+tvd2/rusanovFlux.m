function [numericalFlux,maxSpeed]=rusanovFlux( ...
    physicalFlux,maxWaveSpeed,leftTrace,rightTrace,faceX,time)
%RUSANOVFLUX Local Lax-Friedrichs numerical flux.

leftFlux=physicalFlux(leftTrace,faceX,time);
rightFlux=physicalFlux(rightTrace,faceX,time);
speed=maxWaveSpeed(leftTrace,rightTrace,faceX,time);
if isscalar(speed)
    speed=repmat(abs(speed),1,size(leftTrace,2));
elseif isvector(speed)
    speed=abs(speed(:).');
elseif size(speed,1)>1
    speed=max(abs(speed),[],1);
else
    speed=abs(speed);
end
speed=speed(:).';
if numel(speed)~=size(leftTrace,2) || any(~isfinite(speed)) || any(speed<0)
    error('tvd2:WaveSpeed','MaxWaveSpeed must return a finite speed per face.');
end
if ~isequal(size(leftFlux),size(leftTrace)) ...
        || ~isequal(size(rightFlux),size(rightTrace))
    error('tvd2:PhysicalFlux','Physical flux must preserve the trace size.');
end
numericalFlux=.5*(leftFlux+rightFlux-speed.*(rightTrace-leftTrace));
maxSpeed=max(speed,[],'all');
end
