function [leftTrace,rightTrace,info]=reconstruct( ...
    cellAverage,mesh,boundaries,time,options)
%RECONSTRUCT MUSCL traces at all physical faces.
% CELLAVERAGE is nVariable-by-nCell. Traces are nVariable-by-(nCell+1).

if nargin<5,options=struct();end
if ~isfield(options,'Order'),options.Order=2;end
if ~isfield(options,'Limiter'),options.Limiter="minmod";end
if ~isfield(options,'RuntimeChecks'),options.RuntimeChecks=false;end
if options.RuntimeChecks
    validateattributes(cellAverage,{'numeric'},{'2d','real','finite'});
end
if size(cellAverage,2)~=mesh.CellCount
    error('tvd2:StateSize','State columns must equal mesh.CellCount.');
end
if options.RuntimeChecks && (~isstruct(boundaries) || ~isfield(boundaries,'Left') ...
        || ~isfield(boundaries,'Right'))
    error('tvd2:BoundaryMissing', ...
        'Both boundaries.Left and boundaries.Right must be specified.');
end
if options.Order~=1 && options.Order~=2
    error('tvd2:Order','Order must be 1 or 2.');
end

[extendedState,extendedX,boundaryInfo]=extendState( ...
    cellAverage,mesh,boundaries,time);
slope=zeros(size(extendedState),'like',extendedState);
effectiveOrder=options.Order;
backward=[];
forward=[];
if mesh.CellCount>=2
    backward=(extendedState(:,2:end-1)-extendedState(:,1:end-2)) ./ ...
        (extendedX(2:end-1)-extendedX(1:end-2));
    forward=(extendedState(:,3:end)-extendedState(:,2:end-1)) ./ ...
        (extendedX(3:end)-extendedX(2:end-1));
end
if options.Order==2 && mesh.CellCount>=2
    slope(:,2:end-1)=tvd2.limitedSlope( ...
        backward,forward,options.Limiter);
elseif options.Order==2
    effectiveOrder=1;
end

leftCells=2:(size(extendedState,2)-2);
rightCells=leftCells+1;
faceX=mesh.FaceX;
leftTrace=extendedState(:,leftCells)+slope(:,leftCells).* ...
    (faceX-extendedX(leftCells));
rightTrace=extendedState(:,rightCells)+slope(:,rightCells).* ...
    (faceX-extendedX(rightCells));
unclampedLeft=leftTrace;unclampedRight=rightTrace;
% Geometry-aware face limiting. On stretched meshes the usual named
% limiter formulas alone do not account for unequal center-to-face
% distances. Clamp each trace to the two adjacent cell averages.
faceMinimum=min(extendedState(:,leftCells),extendedState(:,rightCells));
faceMaximum=max(extendedState(:,leftCells),extendedState(:,rightCells));
leftTrace=min(max(leftTrace,faceMinimum),faceMaximum);
rightTrace=min(max(rightTrace,faceMinimum),faceMaximum);
clampTolerance=100*eps(max(1,max(abs(extendedState),[],'all')));
geometryClampFraction=mean([abs(leftTrace-unclampedLeft)>clampTolerance, ...
    abs(rightTrace-unclampedRight)>clampTolerance],'all');
% A Dirichlet value is a face state, not merely a recipe for a ghost-cell
% average.  Preserve the mirror ghosts for slope construction, then impose
% the exterior trace exactly at the physical boundary.
if ~isempty(boundaryInfo.LeftFaceValue)
    leftTrace(:,1)=boundaryInfo.LeftFaceValue;
end
if ~isempty(boundaryInfo.RightFaceValue)
    rightTrace(:,end)=boundaryInfo.RightFaceValue;
end

unlimited=zeros(size(slope),'like',slope);
if mesh.CellCount>=2
    unlimited(:,2:end-1)=.5*(backward+forward);
end
active=abs(unlimited)>100*eps(max(1,max(abs(extendedState),[],'all')));
limiterApplied=options.Order==2 && mesh.CellCount>=2;
if limiterApplied && any(active,'all')
    limiterReduction=mean(abs(slope(active))./abs(unlimited(active)));
    limiterZeroFraction=mean(slope(active)==0);
elseif limiterApplied
    limiterReduction=1;
    limiterZeroFraction=0;
else
    limiterReduction=NaN;
    limiterZeroFraction=NaN;
end
info=struct('RequestedOrder',options.Order, ...
    'EffectiveOrder',effectiveOrder,'Limiter',string(options.Limiter), ...
    'LimiterApplied',limiterApplied, ...
    'LimiterReduction',limiterReduction, ...
    'LimiterZeroFraction',limiterZeroFraction, ...
    'GeometryClampFraction',geometryClampFraction, ...
    'Boundary',boundaryInfo);
end

function [state,x,info]=extendState(U,mesh,boundaries,time)
n=mesh.CellCount;
if n>=2
    leftX=2*mesh.FaceX(1)-mesh.CellX(2:-1:1);
    rightX=2*mesh.FaceX(end)-mesh.CellX(end:-1:end-1);
else
    halfWidth=mesh.CellWidth(1)/2;
    leftX=mesh.FaceX(1)-[3,1]*halfWidth;
    rightX=mesh.FaceX(end)+[1,3]*halfWidth;
end

leftType=lower(string(boundaries.Left.Type));
rightType=lower(string(boundaries.Right.Type));
if xor(leftType=="periodic",rightType=="periodic")
    error('tvd2:PeriodicPair','Periodic boundaries must be specified in pairs.');
end
if leftType=="periodic"
    if n<2
        leftGhost=repmat(U(:,end),1,2);
        rightGhost=repmat(U(:,1),1,2);
    else
        leftGhost=U(:,end-1:end);
        rightGhost=U(:,1:2);
        leftX=mesh.CellX(end-1:end)-mesh.DomainLength;
        rightX=mesh.CellX(1:2)+mesh.DomainLength;
    end
else
    [leftGhost,leftFaceValue]=makeSideGhost( ...
        U,mesh,boundaries.Left,time,"left",leftX);
    [rightGhost,rightFaceValue]=makeSideGhost( ...
        U,mesh,boundaries.Right,time,"right",rightX);
end
if leftType=="periodic"
    leftFaceValue=[];rightFaceValue=[];
end
state=[leftGhost,U,rightGhost];
x=[leftX,mesh.CellX,rightX];
info=struct('Left',leftType,'Right',rightType, ...
    'LeftFaceValue',leftFaceValue,'RightFaceValue',rightFaceValue);
end

function [ghost,faceValue]=makeSideGhost( ...
    U,mesh,specification,time,side,ghostX)
type=lower(string(specification.Type));
n=size(U,2);
faceValue=[];
if side=="left"
    interiorIndex=min([2,1],n);
    interiorForMirror=U(:,interiorIndex);
    interiorX=mesh.CellX(interiorIndex);
else
    % Ghost coordinates are ordered from the boundary outward.  Pair them
    % with physical cells ordered from the boundary inward.
    interiorIndex=max([n,n-1],1);
    interiorForMirror=U(:,interiorIndex);
    interiorX=mesh.CellX(interiorIndex);
end
switch type
    case "dirichlet"
        value=evaluateValue(specification.Value,time,U,mesh,side);
        faceValue=value;
        ghost=2*value-interiorForMirror;
    case "outflow"
        if n<2
            ghost=repmat(U(:,1),1,2);
        elseif side=="left"
            gradient=(U(:,2)-U(:,1))/(mesh.CellX(2)-mesh.CellX(1));
            ghost=U(:,1)+gradient.*(ghostX-mesh.CellX(1));
        else
            gradient=(U(:,end)-U(:,end-1))/ ...
                (mesh.CellX(end)-mesh.CellX(end-1));
            ghost=U(:,end)+gradient.*(ghostX-mesh.CellX(end));
        end
    case "zero-gradient"
        if side=="left",ghost=repmat(U(:,1),1,2);
        else,ghost=repmat(U(:,end),1,2);end
    case "neumann"
        gradient=evaluateValue(specification.Value,time,U,mesh,side);
        ghost=interiorForMirror+gradient.*(ghostX-interiorX);
    otherwise
        error('tvd2:BoundaryType','Unsupported boundary type: %s',type);
end
if size(ghost,2)==1,ghost=repmat(ghost,1,2);end
if ~isequal(size(ghost),[size(U,1),2]) || any(~isfinite(ghost),'all')
    error('tvd2:BoundaryShape', ...
        'Each boundary must generate nVariable-by-2 finite ghost values.');
end
end

function value=evaluateValue(input,time,U,mesh,side)
if isa(input,'function_handle')
    value=input(time,U,mesh,side);
else
    value=input;
end
value=value(:);
if isscalar(value),value=repmat(value,size(U,1),1);end
if numel(value)~=size(U,1) || any(~isfinite(value))
    error('tvd2:BoundaryValue', ...
        'Boundary value must contain one finite value per state variable.');
end
end
