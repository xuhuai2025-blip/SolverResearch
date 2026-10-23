"""稀疏非线性核心的独立数值、合同和原生回退检查。"""
import unittest
import numpy as np
from nonlinearSolver import problemFromFunction,solve
from nonlinearSolver.runtime.gridProgram import GridProgram
from nonlinearSolver.runtime.nativeGrid import libraryInfo


class NonlinearTests(unittest.TestCase):
    """使用普通代数方程和显式依赖图验证求解核心。"""
    def testKnownCoupledRoot(self):
        """耦合非线性方程具有独立指定的已知根。"""
        matrix=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])
        expected=np.array([1.,2.,3.])
        load=matrix@expected+.01*expected**3
        problem=problemFromFunction(lambda x:matrix@x+.01*x**3-load,[.5,1.,1.5])
        state,info=solve(problem,options={'Tolerance':1e-10,'Display':'off'})
        self.assertTrue(info.Converged)
        np.testing.assert_allclose(state,expected,atol=1e-9,rtol=0)
        self.assertLess(info.FinalResidualInf,1e-10)
        self.assertTrue(info.IndependentResidualVerification)
        self.assertGreater(info.NumericalFactorizations,0)
        self.assertEqual(libraryInfo()['abiVersion'],2)

    def testBoundsScalesAndIndependentRuns(self):
        """边界、尺度和每次独立运行时都参与实际求解。"""
        problem=problemFromFunction(lambda x:np.array([x[0]**2-4]),[1.5],
            variableScale=[2.],residualScale=[4.],lowerBound=[.1],upperBound=[4.],isFeasible=lambda x:x[0]>0)
        a,ia=solve(problem,options={'Display':'off','Tolerance':1e-11})
        b,ib=solve(problem,options={'Display':'off','Tolerance':1e-11})
        self.assertTrue(ia.Converged and ib.Converged)
        np.testing.assert_array_equal(a,b)
        self.assertLess(abs(a[0]-2.),1e-10)
        self.assertIsNot(problem.createRuntime(),problem.createRuntime())

    def testCallbackAndOptionsErrors(self):
        """程序异常不能被当作数值不收敛或成功吞掉。"""
        for function in (lambda x:np.zeros(2),lambda x:np.array([1j])):
            with self.assertRaisesRegex(ValueError,'ResidualContract'):
                solve(problemFromFunction(function,[1.]),options={'Display':'off'})
        def broken(x):
            raise RuntimeError('用户程序错误')
        with self.assertRaisesRegex(RuntimeError,'用户程序错误'):
            solve(problemFromFunction(broken,[1.]),options={'Display':'off'})
        for value in (np.nan,np.inf,0.,-1.):
            with self.assertRaises(ValueError):
                solve(problemFromFunction(lambda x:x,[1.]),options={'Display':'off','Tolerance':value})
        with self.assertRaises(InterruptedError):
            solve(problemFromFunction(lambda x:x,[1.]),options={'Display':'off','CancellationCheck':lambda:True})

    def testNativeDependencyInvalidation(self):
        """只改一行的一个分量，不应重算无关行。"""
        program=GridProgram('依赖失效检查')
        values=np.arange(14,dtype=float).reshape(2,7)
        x=program.root('x',values)
        program.announce(dict(value=x**2+program.mean(x)))
        first=program.run(dict(x=values))['value']
        before=program.native.stats()['scalar_values_calculated']
        changed=values.copy()
        changed[0,3]+=2
        second=program.run(dict(x=changed))['value']
        after=program.native.stats()['scalar_values_calculated']
        np.testing.assert_array_equal(first,values**2+np.mean(values,axis=1,keepdims=True))
        np.testing.assert_array_equal(second,changed**2+np.mean(changed,axis=1,keepdims=True))
        np.testing.assert_array_equal(first[1],second[1])
        self.assertLess(after-before,before)

    def testTransactionRestoresRejectedState(self):
        """试探事务结束后恢复缓存和接受态。"""
        problem=problemFromFunction(lambda x:x*x-4,[1.5])
        runtime=problem.createRuntime()
        before=runtime.accept(np.array([1.5])).copy()
        with runtime.transaction(np.array([3.])):
            np.testing.assert_array_equal(runtime.residual(),[5.])
        np.testing.assert_array_equal(runtime.residual(),before)
        np.testing.assert_array_equal(runtime.state,[1.5])


if __name__=='__main__':
    unittest.main()
