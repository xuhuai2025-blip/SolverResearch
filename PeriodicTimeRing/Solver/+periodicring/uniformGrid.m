function grid=uniformGrid(nodeCount,period,origin)
%UNIFORMGRID Construct an odd, uniform Fourier time ring.

if nargin<2 || isempty(period),period=2*pi;end
if nargin<3 || isempty(origin),origin=0;end
validateattributes(nodeCount,{'numeric'}, ...
    {'scalar','integer','>=',3,'finite'});
validateattributes(period,{'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(origin,{'numeric'}, ...
    {'scalar','real','finite'});
if mod(nodeCount,2)~=1
    error('periodicring:EvenNodeCount', ...
        'PeriodicTimeRing requires an odd node count.');
end

modeCount=(nodeCount-1)/2;
phaseStep=2*pi/nodeCount;
phase=(0:nodeCount-1).'*phaseStep;

% Determine the backward-sampling coefficients by making every resolvable
% sine/cosine mode exact.  This is algebraically the Fourier collocation
% differentiation matrix, written in the sampling-coefficient form.
offset=1:(2*modeCount);
systemMatrix=zeros(2*modeCount);
target=zeros(2*modeCount,1);
for mode=1:modeCount
    systemMatrix(2*mode-1,:)=sin(mode*offset*phaseStep);
    systemMatrix(2*mode,:)=1-cos(mode*offset*phaseStep);
    target(2*mode-1)=-mode;
end
betaTheta=zeros(nodeCount,1);
betaTheta(2:end)=systemMatrix\target;
betaTheta(1)=-sum(betaTheta(2:end));

dTheta=zeros(nodeCount);
for row=1:nodeCount
    columns=mod((row-1)-(0:2*modeCount),nodeCount)+1;
    dTheta(row,columns)=betaTheta.';
end
angularRate=2*pi/period;
grid=struct('NodeCount',nodeCount,'Period',period,'Origin',origin, ...
    'Phase',phase,'Time',origin+phase/angularRate, ...
    'D',sparse(angularRate*dTheta), ...
    'Weight',repmat(period/nodeCount,nodeCount,1), ...
    'BetaTheta',betaTheta,'ModeCount',modeCount);
end

