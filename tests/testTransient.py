"""瞬态积分：解析解、原接口测试问题、约束和异常的独立验证。"""
import unittest
import numpy as np
from scipy import sparse
from dasslSolver import TransientProblem, TransientSolver, solve, validateProblem


class TransientTests(unittest.TestCase):
    """检查刚性 ODE、残差 DAE 和奇异质量矩阵，不依赖领域模型。"""
    def assertAnalytic(self,result,exact,tolerance):
        """检查全输出轨迹而非只看终点。"""
        self.assertLess(float(np.max(abs(result.State-exact))),tolerance)
        self.assertTrue(np.isfinite(result.Derivative).all())

    def testStiffExplicitCosine(self):
        """解析刚性方程，在 51 个输出点核验状态和终态残差。"""
        problem=TransientProblem(RHS=lambda t,y:-1000*(y-np.cos(t))-np.sin(t))
        result=solve(problem,np.linspace(0,1,51),[1.],{'RelTol':2e-7,'AbsTol':1e-9})
        self.assertAnalytic(result,np.cos(result.Time)[None,:],2e-5)
        self.assertLess(result.FinalResidualInf,2e-3)
        self.assertEqual(result.ProblemForm,'explicit')
        self.assertGreater(result.LinearFactorizations,0)

    def testResidualAndFixedConsistentInitialConditions(self):
        """修正代数初值同时固定微分状态和指定导数。"""
        problem=TransientProblem(Residual=lambda t,y,yp:np.array([yp[0]+y[0],y[1]-y[0]**2]),
            FixedY0=np.array([True,False]),FixedYP0=np.array([False,True]))
        result=solve(problem,np.linspace(0,1,41),[1.,.8],{'RelTol':1e-7,'AbsTol':[1e-9,1e-9]},[-1.,-2.])
        np.testing.assert_allclose(result.InitialState,[1.,1.],atol=1e-9)
        np.testing.assert_array_equal(result.InputInitialState,[1.,.8])
        np.testing.assert_array_equal(result.InputInitialDerivative,[-1.,-2.])
        self.assertAnalytic(result,np.array([np.exp(-result.Time),np.exp(-2*result.Time)]),3e-5)
        self.assertLess(result.FinalResidualInf,2e-5)

    def testSingularMassMatrix(self):
        """奇异质量矩阵不求逆，保持代数约束。"""
        problem=TransientProblem(RHS=lambda t,y:np.array([-y[0],y[1]-y[0]**2]),
            Mass=np.diag([1.,0.]),MassSingular='yes')
        result=solve(problem,np.linspace(0,1,41),[1.,1.],{'RelTol':1e-7,'AbsTol':[1e-9,1e-9]},[-1.,-2.])
        self.assertAnalytic(result,np.array([np.exp(-result.Time),np.exp(-2*result.Time)]),3e-5)
        self.assertLess(result.FinalResidualInf,3e-4)
        self.assertLess(float(np.max(abs(result.State[1]-result.State[0]**2))),3e-6)

    def testTimeAndStateDependentMass(self):
        """分别验证 M(t)、M(t,y) 及状态相关质量矩阵的雅可比。"""
        problems=[TransientProblem(RHS=lambda t,y:-(1+t)*y,Mass=lambda t:1+t,MassStateDependence='none'),
            TransientProblem(RHS=lambda t,y:-(1+y)*y,Mass=lambda t,y:1+y[0],
                Jacobian=lambda t,y:np.array([[-1-2*y[0]]]),JacobianPattern=np.ones((1,1)),MassVectorPattern=np.ones((1,1)))]
        for problem in problems:
            with self.subTest(form=problem.MassStateDependence):
                result=solve(problem,[0.,1.],[1.],{'RelTol':1e-8,'AbsTol':1e-10})
                self.assertLess(abs(result.FinalState[0]-np.exp(-1)),2e-6)
                self.assertLess(result.FinalResidualInf,2e-5)

    def testSparseAndPartialJacobians(self):
        """隐式解析稀疏矩阵可用于一致初值，缺失部分按差分补齐。"""
        for pair in ((None,sparse.eye(1)),(sparse.eye(1),sparse.eye(1))):
            problem=TransientProblem(Residual=lambda t,y,yp:yp+y,Jacobian=pair,
                JacobianPattern=(sparse.eye(1),None),FixedY0=np.array([True]))
            report=validateProblem(problem,[1.],[-1.])
            self.assertTrue(report.Passed)
            result=solve(problem,[0.,1.],[1.],{'RelTol':1e-8,'AbsTol':1e-10},[-.5])
            self.assertLess(abs(result.FinalState[0]-np.exp(-1)),2e-6)

    def testRhsSparseJacobianAndPattern(self):
        """显式右端的雅可比和模式转为残差形式。"""
        problem=TransientProblem(RHS=lambda t,y:-y,Jacobian=-sparse.eye(2),JacobianPattern=sparse.eye(2))
        result=solve(problem,[0.,1.],[1.,2.],{'RelTol':1e-8,'AbsTol':1e-10})
        np.testing.assert_allclose(result.FinalState,np.array([1.,2.])*np.exp(-1),rtol=2e-6)

    def testInitialConditionsCanBePreservedOrRejected(self):
        """关闭初值计算时拒绝不一致初值，不更改用户固定的分量。"""
        problem=TransientProblem(Residual=lambda t,y,yp:yp+y)
        options={'ComputeConsistentInitialConditions':False}
        with self.assertRaisesRegex(ValueError,'InconsistentInitialConditions'):
            solve(problem,[0.,1.],[1.],options,[0.])
        result=solve(problem,[0.,.1],[1.],options,[-1.])
        self.assertLess(abs(result.FinalState[0]-np.exp(-.1)),2e-5)

    def testEventsAndContinuousOutput(self):
        """事件在非输出网格点终止，输出及连续查询均不越过事件。"""
        problem=TransientProblem(RHS=lambda t,y:-y,Events=lambda t,y:([y[0]-.5],[True],[-1]))
        result=solve(problem,[0.,.2,1.],[1.],{'RelTol':1e-8,'AbsTol':1e-10})
        self.assertEqual(result.EndReason,'event')
        self.assertLess(abs(result.Time[-1]-np.log(2)),2e-6)
        y,yp=result.Evaluate(np.array([.1,.3]))
        np.testing.assert_allclose(y[0],np.exp(-np.array([.1,.3])),rtol=2e-6)
        self.assertEqual(yp.shape,y.shape)
        with self.assertRaisesRegex(ValueError,'EvaluateTime'):
            result.Evaluate(.9)

    def testImplicitEvents(self):
        """隐式事件接收状态导数。"""
        problem=TransientProblem(Residual=lambda t,y,yp:yp+y,Events=lambda t,y,yp:([yp[0]+.5],[True],[1]),
            FixedY0=np.array([True]))
        result=solve(problem,[0.,1.],[1.],{'RelTol':1e-8,'AbsTol':1e-10},[-1.])
        self.assertEqual(result.EndReason,'event')
        self.assertLess(abs(result.Time[-1]-np.log(2)),2e-5)

    def testOutputCallbackStopAndInputCopies(self):
        """进度回调只能修改副本，可在已接受步停止积分。"""
        flags=[]
        def output(t,y,flag):
            flags.append(flag)
            y[:]=1000
            return flag=='step' and t>.25
        result=solve(TransientProblem(RHS=lambda t,y:-y,OutputFcn=output),[0.,1.],[1.])
        self.assertEqual(result.EndReason,'output_callback')
        self.assertLess(result.Time[-1],1.)
        self.assertLess(abs(result.FinalState[0]-np.exp(-result.Time[-1])),2e-5)
        self.assertEqual(flags[0],'init')
        self.assertEqual(flags[-1],'done')

    def testOrderStepControlsAndNdf(self):
        """阶数上限、最大步长及 NDF 模式实际参与积分。"""
        for bdf in (True,False):
            result=solve(TransientProblem(RHS=lambda t,y:-y),[0.,.2],[1.],
                {'MaxOrder':2,'MaxStep':.01,'InitialStep':.001,'BDF':bdf,'RelTol':1e-6,'AbsTol':1e-8})
            self.assertLessEqual(int(result.AcceptedStepOrders.max()),2)
            self.assertLessEqual(np.max(np.diff(np.r_[0.,result.AcceptedStepTimes])),.01+1e-15)
            self.assertLess(abs(result.FinalState[0]-np.exp(-.2)),2e-5)

    def testRobertsonIndex1Conservation(self):
        """强刚性反应的 index-1 形式核验质量守恒及慢变量。"""
        def residual(t,y,yp):
            a,b,c=y
            return np.array([yp[0]+.04*a-1e4*b*c,yp[1]-.04*a+1e4*b*c+3e7*b*b,a+b+c-1])
        problem=TransientProblem(Residual=residual,FixedY0=np.array([True,True,False]),
            FixedYP0=np.array([False,False,True]))
        result=solve(problem,np.r_[0.,np.logspace(-6,2,30)],[1.,0.,0.],
            {'RelTol':1e-7,'AbsTol':[1e-10,1e-12,1e-10]},[-.04,.04,0.])
        self.assertLess(float(np.max(abs(result.State.sum(axis=0)-1))),2e-8)
        self.assertLess(abs(result.FinalState[0]-.61723488),2e-5)
        self.assertLess(result.FinalResidualInf,2e-6)

    def testValidationUsesActualTime(self):
        """预检查在指定时间调用方程，不假设起始时刻为零。"""
        report=validateProblem(TransientProblem(RHS=lambda t,y:-y/t),[1.],sampleTime=1.)
        self.assertEqual(report.SampleTime,1.)
        self.assertEqual(report.SampleCallbackInf,1.)

    def testInvalidContractsAndUnknownOptions(self):
        """方程形状、非有限输出、初值掩码和未知设置明确报错。"""
        with self.assertRaisesRegex(ValueError,'UnknownOption'):
            TransientSolver({'RelToll':1e-6})
        for options in ({'RelTol':np.inf},{'AbsTol':0},{'MaxOrder':6},{'MaxStep':0}):
            with self.assertRaises(ValueError):
                solve(TransientProblem(RHS=lambda t,y:-y),[0.,1.],[1.],options)
        for rhs in (lambda t,y:np.zeros(2),lambda t,y:np.array([np.nan]),lambda t,y:np.array([1j])):
            with self.assertRaises(ValueError):
                solve(TransientProblem(RHS=rhs),[0.,1.],[1.])
        with self.assertRaisesRegex(ValueError,'StateDomain'):
            solve(TransientProblem(RHS=lambda t,y:-y,ValidateState=lambda t,y:y>0),[0.,1.],[-1.])
        with self.assertRaisesRegex(ValueError,'JacobianPattern'):
            solve(TransientProblem(Residual=lambda t,y,yp:yp+y,JacobianPattern=np.eye(1)),[0.,1.],[1.],initialDerivative=[-1.])
        with self.assertRaisesRegex(ValueError,'NonNegativeUnsupported'):
            solve(TransientProblem(RHS=lambda t,y:-y,Mass=1.,NonNegative=[0]),[0.,1.],[1.])
        with self.assertRaisesRegex(ValueError,'InitialMask'):
            solve(TransientProblem(Residual=lambda t,y,yp:yp+y,FixedY0=[1]),[0.,1.],[1.],initialDerivative=[-1.])

    def testOutputTimesAndScalarShape(self):
        """支持重复输出时间，单状态标量查询仍返回一维状态。"""
        problem=TransientProblem(RHS=lambda t,y:-y)
        result=solve(problem,[1.,2.],[1.],{'OutputTimes':[1.,1.5,1.5],'RelTol':1e-8,'AbsTol':1e-10})
        np.testing.assert_array_equal(result.Time,[1.,1.5,1.5,2.])
        self.assertEqual(result.InitialState.shape,(1,))
        self.assertEqual(result.Evaluate(1.2)[0].shape,(1,))
        self.assertEqual(result.Evaluate([])[0].shape,(1,0))
        for times in ([0.,1.],[1.5,1.2],[2.1]):
            with self.assertRaisesRegex(ValueError,'OutputTimes'):
                solve(problem,[1.,2.],[1.],{'OutputTimes':times})

    def testSparseResidualPatternOnly(self):
        """没有解析雅可比时，两个稀疏模式用于后端差分。"""
        problem=TransientProblem(Residual=lambda t,y,yp:yp+y,
            JacobianPattern=(sparse.eye(2),sparse.eye(2)),FixedY0=np.array([True,True]))
        result=solve(problem,[0.,.5],[1.,2.],{'RelTol':1e-8,'AbsTol':1e-10},[-1.,-2.])
        np.testing.assert_allclose(result.FinalState,np.array([1.,2.])*np.exp(-.5),rtol=2e-6)

    def testNonnegativeAndStepBudget(self):
        """负接受态和步数预算按明确错误终止，不返回伪成功。"""
        with self.assertRaisesRegex(ValueError,'NonNegative'):
            solve(TransientProblem(RHS=lambda t,y:np.array([-1.]),NonNegative=[0]),[0.,1.],[.1])
        with self.assertRaisesRegex(RuntimeError,'MaxSteps'):
            solve(TransientProblem(RHS=lambda t,y:-y),[0.,1.],[1.],{'MaxStep':.01,'MaxSteps':1})

    def testUserExceptionsPropagate(self):
        """用户程序错误不能被当成积分收敛或可恢复状态。"""
        def broken(t,y):
            raise RuntimeError('用户回调发生错误')
        with self.assertRaisesRegex(RuntimeError,'用户回调发生错误'):
            solve(TransientProblem(RHS=broken),[0.,1.],[1.])


if __name__=='__main__':
    unittest.main()
