"""可选的回调合同检查和诊断入口，不在导入时启动积分。"""
import numpy as np
from dasslSolver.problem import asProblem, vector
from dasslSolver.evaluator import Evaluator
from dasslSolver.solver import TransientSolver, IntegrationResult


def validateProblem(problem,initialState,initialDerivative=None,sampleTime=0.):
    """在指定初始时间检查方程、矩阵和模式，不修改输入初值。"""
    if not np.isscalar(sampleTime) or not np.isfinite(sampleTime):
        raise ValueError('dassl:SampleTime: 检查时间必须为有限实数标量')
    problem=asProblem(problem)
    state=vector(initialState,'初始状态')
    derivative=np.zeros_like(state) if initialDerivative is None else vector(initialDerivative,'初始导数',state.size)
    evaluator=Evaluator(problem,state.size)
    value=evaluator.residual(float(sampleTime),state,derivative)
    evaluator.patterns(sampleTime,state)
    if problem.Jacobian is not None:
        evaluator.jacobian(sampleTime,state,derivative)
    # 初值一致性与结构合同分别报告；本入口不替调用方调整初值。
    return IntegrationResult(Passed=True,ProblemForm=problem.form,SampleTime=float(sampleTime),
        StateSize=state.size,SampleCallbackInf=float(np.linalg.norm(value,np.inf)))


def debug(problem,timeSpan,initialState,initialDerivative=None,options=None):
    """显式调用时才进行预检查和带统计的积分。"""
    time=vector(timeSpan,'时间区间')
    report=validateProblem(problem,initialState,initialDerivative,float(time[0]))
    settings={} if options is None else dict(options)
    settings['Stats']=True
    result=TransientSolver(settings).solve(problem,timeSpan,initialState,initialDerivative)
    return IntegrationResult(Validation=report,Result=result)
