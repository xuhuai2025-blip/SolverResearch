classdef TestDASSLTransient < matlab.unittest.TestCase
    methods(Test)
        function stiffExplicitProblem(testCase)
            problem.RHS=@(t,y)-1000*(y-cos(t))-sin(t);
            solver=dassl.TransientSolver(struct('Backend',"ode15s", ...
                'RelTol',2e-7,'AbsTol',1e-9,'BDF',true));
            result=solver.solve(problem,linspace(0,1,51),1);
            testCase.verifyLessThan(max(abs(result.State-cos(result.Time)),[],'all'),2e-5);
            testCase.verifyEqual(result.Backend,"ode15s");
            testCase.verifyLessThan(result.FinalResidualInf,2e-3);
        end

        function singularMassMatrixProblem(testCase)
            problem.RHS=@(~,y)[-y(1);y(2)-y(1)^2];
            problem.Mass=diag([1,0]);
            problem.MassSingular="yes";
            solver=dassl.TransientSolver(struct('Backend',"ode15s", ...
                'RelTol',1e-7,'AbsTol',[1e-9;1e-9]));
            result=solver.solve(problem,linspace(0,1,41),[1;1],[-1;-2]);
            exact=[exp(-result.Time);exp(-2*result.Time)];
            testCase.verifyLessThan(max(abs(result.State-exact),[],'all'),3e-5);
            testCase.verifyLessThan(result.FinalResidualInf,3e-4);
        end

        function fullyImplicitResidualProblem(testCase)
            problem.Residual=@(~,y,yp)[yp(1)+y(1);y(2)-y(1)^2];
            problem.FixedY0=[true;false];
            problem.FixedYP0=[false;true];
            solver=dassl.TransientSolver(struct('Backend',"ode15i", ...
                'RelTol',1e-7,'AbsTol',[1e-9;1e-9]));
            result=solver.solve(problem,linspace(0,1,41),[1;.8],[-1;-2]);
            exact=[exp(-result.Time);exp(-2*result.Time)];
            testCase.verifyEqual(result.InitialState,[1;1],'AbsTol',1e-9);
            testCase.verifyLessThan(max(abs(result.State-exact),[],'all'),3e-5);
            testCase.verifyLessThan(result.FinalResidualInf,2e-5);
        end

        function invalidStateIsRejected(testCase)
            problem.RHS=@(~,y)-y;
            problem.ValidateState=@(~,y)y>0;
            solver=dassl.TransientSolver();
            testCase.verifyError(@()solver.solve(problem,[0,1],-1), ...
                'dassl:StateDomain');
        end

        function timeOnlyMassFunctionIsAudited(testCase)
            problem.RHS=@(~,y)-y;
            problem.Mass=@(~)1;
            problem.MassStateDependence="none";
            solver=dassl.TransientSolver(struct('Backend',"ode15s", ...
                'RelTol',1e-8,'AbsTol',1e-10));
            result=solver.solve(problem,[0,1],1);
            testCase.verifyLessThan(abs(result.FinalState-exp(-1)),2e-6);
            testCase.verifyLessThan(result.FinalResidualInf,2e-5);
        end

        function reportsInputAndSolverAcceptedInitialData(testCase)
            problem.RHS=@(~,y)[-y(1);y(2)-y(1)^2];
            problem.Mass=diag([1,0]);
            problem.MassSingular="yes";
            solver=dassl.TransientSolver(struct('Backend',"ode15s"));
            result=solver.solve(problem,[0,.1],[1;.8],[-1;-2]);
            testCase.verifyEqual(result.InputInitialState,[1;.8]);
            testCase.verifyEqual(result.InputInitialDerivative,[-1;-2]);
            testCase.verifyEqual(result.InitialState,result.State(:,1), ...
                'AbsTol',1e-13);
            testCase.verifyEqual(result.InitialDerivative, ...
                result.Derivative(:,1),'AbsTol',1e-13);
            testCase.verifyEqual(result.InitialState(2),1,'AbsTol',1e-8);
        end

        function functionalShortcutWorks(testCase)
            problem.RHS=@(~,y)-y;
            result=dassl.solve(problem,[0,1],1, ...
                struct('Backend',"ode15s",'RelTol',1e-8,'AbsTol',1e-10));
            testCase.verifyLessThan(abs(result.FinalState-exp(-1)),2e-6);
            testCase.verifyGreaterThan(result.TotalElapsedSeconds,0);
            testCase.verifyEqual(result.ElapsedSeconds,result.TotalElapsedSeconds);

            dae.Residual=@(~,y,yp)yp+y;
            dae.FixedY0=true;dae.FixedYP0=false;
            implicit=dassl.solve(dae,[0,.1],1,[],-1);
            testCase.verifyEqual(implicit.Backend,"ode15i");

            dae.FixedY0=[];dae.FixedYP0=[];
            implicit=dassl.solve(dae,[0,.1],1,[],-1);
            testCase.verifyEqual(implicit.Backend,"ode15i");
        end

        function backendSpecificContractsHaveClearErrors(testCase)
            residualProblem.Residual=@(~,y,yp)yp+y;
            residualProblem.JacobianPattern=speye(1);
            solver=dassl.TransientSolver(struct('Backend',"ode15i"));
            testCase.verifyError(@()solver.solve( ...
                residualProblem,[0,1],1,-1),'dassl:JacobianPattern');

            massProblem.RHS=@(~,y)-y;
            massProblem.Mass=1;
            massProblem.NonNegative=1;
            solver=dassl.TransientSolver(struct('Backend',"ode15s"));
            testCase.verifyError(@()solver.solve( ...
                massProblem,[0,1],1),'dassl:NonNegativeUnsupported');
        end

        function optionalValidatorUsesActualInitialTimeAndAuto(testCase)
            problem.RHS=@(t,y)-y/t;
            problem.Residual=[];
            report=dassltools.validateProblem(problem,1,[],"auto",1);
            testCase.verifyEqual(report.Backend,"ode15s");
            testCase.verifyEqual(report.SampleTime,1);
            testCase.verifyEqual(report.SampleCallbackInf,1);

            dae.Residual=@(~,y,yp)yp+y;
            report=dassltools.validateProblem(dae,1,[],"auto",2);
            testCase.verifyEqual(report.Backend,"ode15i");
        end

        function optionalDebugLayerRunsOnlyWhenCalled(testCase)
            problem.RHS=@(t,y)-y/t;
            diagnostics=dassltools.debug(problem,[1,1.05],1,[], ...
                struct('Backend',"auto",'RelTol',1e-7,'AbsTol',1e-9));
            testCase.verifyTrue(diagnostics.Validation.Passed);
            testCase.verifyEqual(diagnostics.Validation.SampleTime,1);
            testCase.verifyLessThan(abs(diagnostics.Result.FinalState-1/1.05), ...
                2e-6);
        end

        function validatorAllowsPartialImplicitJacobians(testCase)
            problem.Residual=@(~,y,yp)yp+y;
            problem.Jacobian={[],1};
            problem.JacobianPattern={1,[]};
            report=dassltools.validateProblem(problem,1,-1,"ode15i",0);
            testCase.verifyTrue(report.Passed);
        end

        function unknownOptionIsRejected(testCase)
            testCase.verifyError(@()dassl.TransientSolver( ...
                struct('RelToll',1e-6)),'dassl:UnknownOption');
        end
    end
end
