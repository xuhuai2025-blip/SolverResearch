function value=divergence(faceFlux,cellMeasure)
%DIVERGENCE Outward conservative flux difference per cell measure.

cellMeasure=cellMeasure(:).';
if size(faceFlux,2)~=numel(cellMeasure)+1 || any(cellMeasure<=0) ...
        || any(~isfinite(cellMeasure))
    error('tvd2:DivergenceSize', ...
        'Cell measure must match faceFlux columns minus one.');
end
value=(faceFlux(:,2:end)-faceFlux(:,1:end-1))./cellMeasure;
end

