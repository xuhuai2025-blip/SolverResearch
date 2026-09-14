function mesh=makeMesh(faceX,cellMeasure)
%MAKEMESH Construct a validated one-dimensional finite-volume mesh.

validateattributes(faceX,{'numeric'},{'vector','real','finite','nonempty'});
faceX=faceX(:).';
if numel(faceX)<2 || any(diff(faceX)<=0)
    error('tvd2:MeshFaces','Face coordinates must be strictly increasing.');
end
width=diff(faceX);
center=.5*(faceX(1:end-1)+faceX(2:end));
if nargin<2 || isempty(cellMeasure)
    cellMeasure=width;
elseif isscalar(cellMeasure)
    cellMeasure=repmat(cellMeasure,1,numel(width));
else
    cellMeasure=cellMeasure(:).';
end
if numel(cellMeasure)~=numel(width) || any(~isfinite(cellMeasure)) ...
        || any(cellMeasure<=0)
    error('tvd2:CellMeasure', ...
        'Cell measure must contain one positive finite value per cell.');
end
mesh=struct('FaceX',faceX,'CellX',center,'CellWidth',width, ...
    'CellMeasure',cellMeasure,'CellCount',numel(width), ...
    'DomainLength',faceX(end)-faceX(1));
end

