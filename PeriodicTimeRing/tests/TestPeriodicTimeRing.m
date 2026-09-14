classdef TestPeriodicTimeRing < matlab.unittest.TestCase
    methods(Test)
        function harmonicsAndQuadratureAreExact(testCase)
            period=3;origin=.2;grid=periodicring.uniformGrid(7,period,origin);
            omega=2*pi/period;phase=grid.Phase.';
            values=[sin(phase);cos(2*phase)];
            exact=[omega*cos(phase);-2*omega*sin(2*phase)];
            testCase.verifyEqual(periodicring.differentiate(values,grid), ...
                exact,'AbsTol',2e-12);
            testCase.verifyEqual(sum(grid.Weight),period,'AbsTol',2e-15);
            testCase.verifyEqual(periodicring.integrate(values,grid), ...
                zeros(2,1),'AbsTol',2e-15);
        end

        function resamplingPreservesComplexValues(testCase)
            grid=periodicring.uniformGrid(7,2*pi,-.4);
            target=linspace(grid.Origin,grid.Origin+grid.Period,19);
            values=exp(2i*grid.Phase).';
            actual=periodicring.resample(values,grid,target);
            exact=exp(2i*(target-grid.Origin));
            testCase.verifyEqual(actual,exact,'AbsTol',2e-12);
        end

        function solvesManufacturedPeriodicOde(testCase)
            period=3;origin=.2;grid=periodicring.uniformGrid(7,period,origin);
            p=struct('Origin',origin,'Omega',2*pi/period);
            problem.StateCount=1;
            problem.LocalResidual=@forcedOde;
            problem.StatePattern=sparse(1);
            problem.DerivativePattern=sparse(1);
            compiled=periodicring.compile(problem,grid);
            result=periodicring.solve(compiled,zeros(1,grid.NodeCount), ...
                struct('Tolerance',1e-11),p);
            exact=sin(p.Omega*(grid.Time-origin)).';
            testCase.verifyTrue(result.Info.Converged);
            testCase.verifyEqual(result.State,exact,'AbsTol',2e-9);
        end

        function solvesIndexOneDae(testCase)
            grid=periodicring.uniformGrid(7,2*pi,0);
            problem.StateCount=2;
            problem.LocalResidual=@indexOneResidual;
            problem.StatePattern=sparse([0,1;1,0]);
            problem.DerivativePattern=sparse([1,0;0,0]);
            compiled=periodicring.compile(problem,grid);
            result=periodicring.solve(compiled,zeros(2,grid.NodeCount), ...
                struct('Tolerance',1e-11));
            exact=[sin(grid.Time).';cos(grid.Time).'];
            testCase.verifyEqual(result.State,exact,'AbsTol',2e-9);
        end

        function globalConstraintRemovesFreeConstant(testCase)
            grid=periodicring.uniformGrid(7,2*pi,0);
            problem.StateCount=1;
            problem.LocalResidual=@freeConstantResidual;
            problem.StatePattern=sparse(0);
            problem.DerivativePattern=sparse(1);
            problem.GlobalRows=periodicring.flatIndex(1,1,1);
            problem.GlobalResidual=@zeroMean;
            problem.GlobalPattern=sparse(ones(1,grid.NodeCount));
            compiled=periodicring.compile(problem,grid);
            result=periodicring.solve(compiled,ones(1,grid.NodeCount), ...
                struct('Tolerance',1e-11));
            testCase.verifyEqual(result.State,sin(grid.Time).','AbsTol',2e-9);
            testCase.verifyEqual(periodicring.integrate(result.State,grid), ...
                0,'AbsTol',2e-10);
        end

        function catchesUnderreportedPattern(testCase)
            grid=periodicring.uniformGrid(5,2*pi,0);
            problem.StateCount=1;
            problem.LocalResidual=@forcedOde;
            problem.StatePattern=sparse(0);
            problem.DerivativePattern=sparse(0);
            compiled=periodicring.compile(problem,grid);
            p=struct('Origin',0,'Omega',1);
            testCase.verifyError(@()periodicringtools.debug(compiled, ...
                zeros(1,grid.NodeCount),p), ...
                'periodicring:PatternUnderreported');
        end

        function rejectsEvenNodeCount(testCase)
            testCase.verifyError(@()periodicring.uniformGrid(6), ...
                'periodicring:EvenNodeCount');
        end

        function optionalValidationAndLayoutAreConsistent(testCase)
            grid=periodicring.uniformGrid(7,3.7,-.4);
            problem.StateCount=2;
            problem.LocalResidual=@indexOneResidual;
            problem.StatePattern=sparse([0,1;1,0]);
            problem.DerivativePattern=sparse([1,0;0,0]);
            report=periodicringtools.validateProblem( ...
                problem,grid,zeros(2,grid.NodeCount),[]);
            sentinel=reshape(1:14,2,7);
            testCase.verifyEqual(reshape(sentinel(:),2,7),sentinel);
            testCase.verifyEqual(periodicring.flatIndex(2,4,2),8);
            testCase.verifyTrue(report.Passed);
            testCase.verifyTrue(report.ColoringValid);
        end

        function rowVectorScalesStayOneDimensional(testCase)
            residual=@(x)x-[1;2;3];
            options=struct('VariableScale',[1,2,3], ...
                'ResidualScale',[3,2,1],'Display',"none");
            [state,info]=periodicring.solveNewton( ...
                residual,zeros(3,1),speye(3),options);
            testCase.verifyEqual(state,[1;2;3],'AbsTol',1e-8);
            testCase.verifyEqual(info.UnknownCount,3);
        end

        function unknownNewtonOptionIsRejected(testCase)
            testCase.verifyError(@()periodicring.solveNewton( ...
                @(x)x,1,1,struct('Tolrance',1e-8)), ...
                'periodicring:UnknownOption');
        end

        function conflictingUserColoringIsRejected(testCase)
            testCase.verifyError(@()periodicring.solveNewton( ...
                @(x)x,[1;2],speye(2)+sparse([1,1;0,0]), ...
                struct('Colors',[1;1])),'periodicring:ColorConflict');
        end
    end
end

function residual=forcedOde(time,state,derivative,p)
phase=p.Omega*(time-p.Origin);
residual=derivative+state-(p.Omega*cos(phase)+sin(phase));
end

function residual=indexOneResidual(time,state,derivative,~)
residual=[derivative(1)-state(2);state(1)-sin(time)];
end

function residual=freeConstantResidual(time,~,derivative,~)
residual=derivative-cos(time);
end

function residual=zeroMean(grid,state,~)
residual=periodicring.integrate(state,grid)/grid.Period;
end
