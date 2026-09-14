function [derivative,info]=spatialOperator(problem,time,state,mesh,options)
%SPATIALOPERATOR Conservative semi-discrete finite-volume operator.

if nargin<5,options=struct();end
if ~isfield(options,'Order'),options.Order=2;end
if ~isfield(options,'Limiter'),options.Limiter="minmod";end
if ~isfield(options,'RuntimeChecks'),options.RuntimeChecks=false;end
[leftTrace,rightTrace,reconstruction]=tvd2.reconstruct( ...
    state,mesh,problem.Boundary,time,options);
if isfield(problem,'NumericalFlux') && ~isempty(problem.NumericalFlux)
    faceFlux=problem.NumericalFlux( ...
        leftTrace,rightTrace,mesh.FaceX,time);
    maxSpeed=NaN;
    if isfield(problem,'MaxWaveSpeed') && ~isempty(problem.MaxWaveSpeed)
        speed=problem.MaxWaveSpeed(leftTrace,rightTrace,mesh.FaceX,time);
        maxSpeed=max(abs(speed),[],'all');
    end
else
    required={'Flux','MaxWaveSpeed'};
    for index=1:numel(required)
        if ~isfield(problem,required{index}) || isempty(problem.(required{index}))
            error('tvd2:Problem','problem.%s is required.',required{index});
        end
    end
    [faceFlux,maxSpeed]=tvd2.rusanovFlux(problem.Flux, ...
        problem.MaxWaveSpeed,leftTrace,rightTrace,mesh.FaceX,time);
end
if ~isequal(size(faceFlux),[size(state,1),mesh.CellCount+1]) ...
        || (options.RuntimeChecks && any(~isfinite(faceFlux),'all'))
    error('tvd2:NumericalFlux', ...
        'Numerical flux must be nVariable-by-(nCell+1) and finite.');
end
derivative=tvd2.netFlux(faceFlux)./mesh.CellMeasure;
source=zeros(size(state),'like',state);
if isfield(problem,'Source') && ~isempty(problem.Source)
    source=problem.Source(state,mesh.CellX,time);
    if ~isequal(size(source),size(state)) ...
            || (options.RuntimeChecks && any(~isfinite(source),'all'))
        error('tvd2:Source','Source must be finite and match the state size.');
    end
    derivative=derivative+source;
end
info=struct('FaceFlux',faceFlux,'LeftTrace',leftTrace, ...
    'RightTrace',rightTrace,'MaxWaveSpeed',maxSpeed, ...
    'Reconstruction',reconstruction,'Source',source);
end
