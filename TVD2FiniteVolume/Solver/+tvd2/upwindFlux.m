function flux=upwindFlux(faceTransport,leftTrace,rightTrace)
%UPWINDFLUX Conservative scalar-transport upwind flux.

if isscalar(faceTransport)
    faceTransport=repmat(faceTransport,1,size(leftTrace,2));
elseif isvector(faceTransport)
    faceTransport=faceTransport(:).';
end
if size(faceTransport,2)~=size(leftTrace,2)
    error('tvd2:FluxSize','One transport value is required per face.');
end
if size(faceTransport,1)==1 && size(leftTrace,1)>1
    faceTransport=repmat(faceTransport,size(leftTrace,1),1);
end
if ~isequal(size(faceTransport),size(leftTrace)) ...
        || ~isequal(size(leftTrace),size(rightTrace))
    error('tvd2:FluxSize','Flux and trace sizes are incompatible.');
end
flux=max(faceTransport,0).*leftTrace+min(faceTransport,0).*rightTrace;
end
