function matrix=evaluationMatrix(grid,targetTime)
%EVALUATIONMATRIX Fourier interpolation matrix from nodes to target times.

validateattributes(targetTime,{'numeric'},{'vector','real','finite'});
targetTime=targetTime(:);
m=grid.ModeCount;
modes=[0:m,-m:-1];
nodeBasis=exp(1i*grid.Phase(:)*modes);
targetPhase=2*pi*(targetTime-grid.Origin)/grid.Period;
targetBasis=exp(1i*targetPhase*modes);
% The cardinal interpolation matrix is real; removing roundoff-level
% imaginary parts still preserves genuinely complex input values.
matrix=real(targetBasis/nodeBasis);
end

