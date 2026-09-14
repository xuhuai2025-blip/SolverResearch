classdef TestTVD2 < matlab.unittest.TestCase
    methods(Test)
        function limitersAreBounded(testCase)
            left=[-2,-1,0,1,2];right=[-1,1,2,3,1];
            value=tvd2.minmod(left,right);
            testCase.verifyEqual(value,[-1,0,0,1,1]);
            for limiter=["minmod","mc","vanleer"]
                slope=tvd2.limitedSlope(left,right,limiter);
                testCase.verifyTrue(all(isfinite(slope)));
                testCase.verifyEqual(slope(2:3),[0,0]);
            end
        end

        function linearFieldReconstructsOnStretchedMesh(testCase)
            mesh=tvd2.makeMesh([0,.04,.15,.37,.68,1]);
            exact=@(x)2.5-1.2*x;
            U=exact(mesh.CellX);
            bc=struct('Left',tvd2.boundary("dirichlet",exact(mesh.FaceX(1))), ...
                'Right',tvd2.boundary("dirichlet",exact(mesh.FaceX(end))));
            [left,right,info]=tvd2.reconstruct(U,mesh,bc,0, ...
                struct('Order',2,'Limiter',"minmod"));
            testCase.verifyEqual(left,exact(mesh.FaceX),'AbsTol',2e-14);
            testCase.verifyEqual(right,exact(mesh.FaceX),'AbsTol',2e-14);
            testCase.verifyEqual(info.EffectiveOrder,2);
        end

        function netFluxIsConservative(testCase)
            rng(4);flux=randn(3,18);measure=.1+rand(1,17);
            net=tvd2.netFlux(flux);
            divergence=tvd2.divergence(flux,measure);
            testCase.verifyEqual(sum(net,2),flux(:,1)-flux(:,end), ...
                'AbsTol',2e-14);
            testCase.verifyEqual(sum(divergence.*measure,2), ...
                flux(:,end)-flux(:,1),'AbsTol',2e-14);
        end

        function nonperiodicAdvectionHasNoNewExtrema(testCase)
            mesh=tvd2.makeMesh(linspace(0,1,81));
            problem.Flux=@(U,x,t)U;
            problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
            problem.Boundary=struct( ...
                'Left',tvd2.boundary("dirichlet",1), ...
                'Right',tvd2.boundary("outflow"));
            result=tvd2.solve(problem,mesh,zeros(1,mesh.CellCount),[0,.25], ...
                struct('Limiter',"minmod",'Integrator',"ssp-rk3",'CFL',.4));
            final=result.State(:,:,end);
            testCase.verifyGreaterThan(mean(final(mesh.CellX<.15)),.98);
            testCase.verifyLessThan(mean(final(mesh.CellX>.45)),1e-3);
            testCase.verifyGreaterThanOrEqual(min(final),-2e-13);
            testCase.verifyLessThanOrEqual(max(final),1+2e-13);
        end

        function smoothAdvectionConverges(testCase)
            errors=zeros(1,2);grids=[50,100];finalTime=.2;
            for index=1:2
                n=grids(index);mesh=tvd2.makeMesh(linspace(0,1,n+1));
                initial=1+.2*sin(2*pi*mesh.CellX);
                problem.Flux=@(U,x,t)U;
                problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
                periodic=tvd2.boundary("periodic");
                problem.Boundary=struct('Left',periodic,'Right',periodic);
                result=tvd2.solve(problem,mesh,initial,[0,finalTime], ...
                    struct('Limiter',"mc",'Integrator',"ssp-rk3",'CFL',.35));
                exact=1+.2*sin(2*pi*mod(mesh.CellX-finalTime,1));
                errors(index)=mean(abs(result.State(:,:,end)-exact));
            end
            observed=log(errors(1)/errors(2))/log(2);
            testCase.verifyGreaterThan(observed,1.7);
        end

        function oneCellFallbackIsExplicit(testCase)
            mesh=tvd2.makeMesh([0,1]);U=2;
            bc=struct('Left',tvd2.boundary("zero-gradient"), ...
                'Right',tvd2.boundary("zero-gradient"));
            [left,right,info]=tvd2.reconstruct(U,mesh,bc,0, ...
                struct('Order',2,'Limiter',"minmod"));
            testCase.verifyEqual(left,[2,2]);
            testCase.verifyEqual(right,[2,2]);
            testCase.verifyEqual(info.EffectiveOrder,1);
        end

        function dirichletFaceStateIsExactAtBothOrders(testCase)
            mesh=tvd2.makeMesh([0,.2,.55,1]);U=[.2,.4,.7];
            bc=struct('Left',tvd2.boundary("dirichlet",1.25), ...
                'Right',tvd2.boundary("dirichlet",.15));
            for order=[1,2]
                [left,right]=tvd2.reconstruct(U,mesh,bc,0, ...
                    struct('Order',order,'Limiter',"minmod"));
                testCase.verifyEqual(left(1),1.25,'AbsTol',1e-15);
                testCase.verifyEqual(right(end),.15,'AbsTol',1e-15);
                transport=[2,zeros(1,mesh.CellCount-1),-3];
                flux=tvd2.upwindFlux(transport,left,right);
                testCase.verifyEqual(flux(1),2*1.25,'AbsTol',1e-15);
                testCase.verifyEqual(flux(end),-3*.15,'AbsTol',1e-15);
            end
        end

        function stretchedGridLimitersRemainBoundedAndTvd(testCase)
            widths=repmat([1,8],1,20);faces=[0,cumsum(widths)];
            faces=faces/faces(end);mesh=tvd2.makeMesh(faces);
            initial=double(mesh.CellX<.5);
            periodic=tvd2.boundary("periodic");
            problem.Flux=@(U,~,~)U;
            problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
            problem.Boundary=struct('Left',periodic,'Right',periodic);
            initialVariation=periodicVariation(initial);
            for limiter=["minmod","mc","vanleer"]
                result=tvd2.solve(problem,mesh,initial,[0,.12], ...
                    struct('Limiter',limiter,'Integrator',"ssp-rk3",'CFL',.35));
                final=result.State(:,:,end);
                testCase.verifyGreaterThanOrEqual(min(final),-2e-12);
                testCase.verifyLessThanOrEqual(max(final),1+2e-12);
                testCase.verifyLessThanOrEqual( ...
                    periodicVariation(final),initialVariation+2e-11);
            end
        end

        function fluxHelpersBroadcastAndOrientVectors(testCase)
            left=[1,2,3];right=[4,5,6];
            testCase.verifyEqual(tvd2.upwindFlux(2,left,right),2*left);
            testCase.verifyEqual(tvd2.upwindFlux(-2,left,right),-2*right);
            testCase.verifyEqual(tvd2.upwindFlux(0,left,right),zeros(1,3));
            flux=@(U,~,~)U;
            speed=@(~,~,~,~)[1;2;3];
            [actual,maxSpeed]=tvd2.rusanovFlux( ...
                flux,speed,left,right,0:2,0);
            expected=.5*(left+right-[1,2,3].*(right-left));
            testCase.verifyEqual(actual,expected);
            testCase.verifyEqual(maxSpeed,3);
        end

        function customMeasureNeedsExplicitStep(testCase)
            mesh=tvd2.makeMesh(linspace(0,1,11),.01);
            periodic=tvd2.boundary("periodic");
            problem.Flux=@(U,~,~)U;
            problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
            problem.Boundary=struct('Left',periodic,'Right',periodic);
            testCase.verifyError(@()tvd2.solve(problem,mesh, ...
                zeros(1,mesh.CellCount),[0,.1]),'tvd2:CFLMeasure');
        end

        function laterStageSpeedTriggersSafeRetry(testCase)
            mesh=tvd2.makeMesh(linspace(0,1,41));
            initial=double(mesh.CellX<.5);
            periodic=tvd2.boundary("periodic");
            speed=@(time)1+1e4*time;
            problem.Flux=@(U,~,time)speed(time)*U;
            problem.MaxWaveSpeed=@(~,~,x,time)speed(time)*ones(size(x));
            problem.Boundary=struct('Left',periodic,'Right',periodic);
            result=tvd2.solve(problem,mesh,initial,[0,.005], ...
                struct('Limiter',"minmod",'Integrator',"ssp-rk3",'CFL',.4));
            final=result.State(:,:,end);
            testCase.verifyGreaterThan(result.RejectedStepCount,0);
            testCase.verifyGreaterThanOrEqual(min(final),-2e-12);
            testCase.verifyLessThanOrEqual(max(final),1+2e-12);
            testCase.verifyLessThanOrEqual( ...
                periodicVariation(final),periodicVariation(initial)+2e-11);
        end

        function stretchedNonperiodicMinmodIsSecondOrder(testCase)
            counts=[25,50,100];errors=zeros(size(counts));finalTime=.08;
            for index=1:numel(counts)
                n=counts(index);faces=(linspace(0,1,n+1)).^1.4;
                mesh=tvd2.makeMesh(faces);
                average=@(time)(exp(faces(2:end)-time) ...
                    -exp(faces(1:end-1)-time))./mesh.CellWidth;
                initial=average(0);
                problem.Flux=@(U,~,~)U;
                problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
                problem.Boundary=struct( ...
                    'Left',tvd2.boundary("dirichlet",@(t,varargin)exp(-t)), ...
                    'Right',tvd2.boundary("outflow"));
                result=tvd2.solve(problem,mesh,initial,[0,finalTime], ...
                    struct('Limiter',"minmod",'Integrator',"ssp-rk3",'CFL',.25));
                errors(index)=mean(abs(result.State(:,:,end)-average(finalTime)));
            end
            observed=log(errors(1:end-1)./errors(2:end))/log(2);
            testCase.verifyGreaterThan(min(observed),1.8);
        end

        function optionalToolsAreExplicitAndOperational(testCase)
            mesh=tvd2.makeMesh(linspace(0,1,11));
            periodic=tvd2.boundary("periodic");
            problem.Flux=@(U,~,~)U;
            problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
            problem.Boundary=struct('Left',periodic,'Right',periodic);
            state=1+.1*sin(2*pi*mesh.CellX);
            validation=tvd2tools.validateProblem( ...
                problem,mesh,state,0,struct('Limiter',"minmod"));
            diagnostics=tvd2tools.debug( ...
                problem,mesh,state,0,struct('Limiter',"minmod"));
            testCase.verifyTrue(validation.Passed);
            testCase.verifyEqual(diagnostics.Validation.Passed,true);
        end

        function unknownOptionIsRejected(testCase)
            mesh=tvd2.makeMesh(linspace(0,1,5));
            periodic=tvd2.boundary("periodic");
            problem.Flux=@(U,~,~)U;
            problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
            problem.Boundary=struct('Left',periodic,'Right',periodic);
            testCase.verifyError(@()tvd2.solve(problem,mesh,zeros(1,4), ...
                [0,.1],struct('CLF',.4)),'tvd2:UnknownOption');
        end
    end
end

function value=periodicVariation(state)
state=state(:).';
value=sum(abs(diff([state,state(1)])));
end
