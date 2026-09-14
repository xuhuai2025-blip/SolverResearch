function results=runDASSLTransientTests()
%RUNDASSLTRANSIENTTESTS Run this library's independent test suite.

root=fileparts(mfilename('fullpath'));
solverRoot=fullfile(root,'Solver');toolsRoot=fullfile(root,'Tools');
oldPath=path;
addpath(solverRoot,toolsRoot);
cleanup=onCleanup(@()path(oldPath));
suite=testsuite(fullfile(root,'tests'),'IncludeSubfolders',true);
results=run(suite);
assert(~isempty(results) && all([results.Passed]),'DASSLTransient:testFailure', ...
    'At least one DASSLTransient test failed.');
end
