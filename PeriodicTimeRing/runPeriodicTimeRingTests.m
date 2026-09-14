function results=runPeriodicTimeRingTests
%RUNPERIODICTIMERINGTESTS Run the self-contained PeriodicTimeRing suite.

root=fileparts(mfilename('fullpath'));
solverRoot=fullfile(root,'Solver');toolsRoot=fullfile(root,'Tools');
oldPath=path;
addpath(solverRoot,toolsRoot);
cleanup=onCleanup(@()path(oldPath));
results=runtests(fullfile(root,'tests'));
assert(~isempty(results),'PeriodicTimeRing:noTests','No tests were discovered.');
assertSuccess(results);
end
