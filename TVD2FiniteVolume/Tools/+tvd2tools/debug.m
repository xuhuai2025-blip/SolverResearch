function diagnostics=debug(problem,mesh,state,time,options)
%DEBUG Evaluate one spatial residual and return detailed diagnostics.

if nargin<4,time=0;end
if nargin<5,options=struct();end
validation=tvd2tools.validateProblem(problem,mesh,state,time,options);
[derivative,details]=tvd2.spatialOperator(problem,time,state,mesh,options);
suggestedStep=NaN;
if isfinite(details.MaxWaveSpeed) && details.MaxWaveSpeed>0
    cfl=.45;if isfield(options,'CFL'),cfl=options.CFL;end
    suggestedStep=cfl*min(mesh.CellWidth)/details.MaxWaveSpeed;
end
diagnostics=struct('Validation',validation,'Derivative',derivative, ...
    'FaceFlux',details.FaceFlux,'LeftTrace',details.LeftTrace, ...
    'RightTrace',details.RightTrace,'Reconstruction',details.Reconstruction, ...
    'SuggestedTimeStep',suggestedStep);
end

