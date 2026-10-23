"""普通残差回调适配：无领域模型时也可直接构建独立数值问题。"""
import numpy as np
from nonlinearSolver.problem import NonlinearProblem
from nonlinearSolver.policies import VariablePolicies
from nonlinearSolver.runtime.gridProgram import GridProgram
from nonlinearSolver.runtime.scalarRuntime import ScalarRuntime
from nonlinearSolver.errors import EvaluationDomainError


# 维护数值状态、依赖或计算接口。
class FunctionEvaluator:
    """完整残差与合法试探使用同一函数，最终验收仍独立计算。"""
    # 初始化数值对象。
    def __init__(self,residual,size,isFeasible):
        self.residual=residual;self.size=size;self.isFeasible=isFeasible

    # 维护数值状态、依赖或计算接口。
    def evaluateResidual(self,state):
        # 不隐式展平矩阵或接受超定问题，避免把接口错误误当成收敛失败。
        value=np.asarray(self.residual(np.asarray(state,float).copy()))
        if value.dtype.kind not in 'iuf' or value.shape!=(self.size,) or not np.isfinite(value).all():
            raise ValueError('nonlinear:ResidualContract: 残差必须是与未知量等长的有限实数向量')
        return np.asarray(value,float)

    # 维护数值状态、依赖或计算接口。
    def evaluateTrial(self,state):
        if self.isFeasible is not None and not self.isFeasible(np.asarray(state,float)):
            return None,False
        return self.evaluateResidual(state),True


# 维护数值状态、依赖或计算接口。
def problemFromFunction(residual,initialState,*,variableScale=None,residualScale=None,
                        lowerBound=None,upperBound=None,isFeasible=None,identity=None):
    """将 F(x)=0 接入唯一算法；黑箱函数保守声明全依赖，不推测零导数。"""
    initial=np.asarray(initialState)
    if initial.dtype.kind not in 'iuf' or initial.ndim!=1 or initial.size==0 or not np.isfinite(initial).all():
        raise ValueError('nonlinear:InitialState: 初值必须是非空有限实数向量')
    initial=np.asarray(initial,float).copy()
    if not callable(residual):raise TypeError('nonlinear:ResidualFunction: 必须提供可调用的残差函数')
    size=initial.size;reference=FunctionEvaluator(residual,size,isFeasible)

    # 维护数值状态、依赖或计算接口。
    def runtimeFactory():
        # 每次工厂调用都拥有独立图、状态、有效位和 journal，多个求解互不共享缓存。
        program=GridProgram('通用残差回调');runtime=ScalarRuntime(initial,program)
        maps=np.tile(np.arange(size),(size,1))
        # 维护数值状态、依赖或计算接口。
        def calculate(rows,unused):
            value,valid=reference.evaluateTrial(runtime.root.value.ravel())
            if not valid:raise EvaluationDomainError('nonlinear:TrialDomain: 试探超出用户定义域')
            return value[rows]
        value=program.op('完整用户残差',[runtime.root],[maps],(size,1),calculate)
        runtime.outputs={'F':value};runtime.outputRows={'F':np.arange(size)}
        program.announce(runtime.outputs);runtime._freeze()
        return runtime

    # 外部可变参数应提供身份回调；默认身份只保证当前适配对象内部一致。
    problemIdentity=identity if identity is not None else lambda:('function-problem',id(reference))
    return NonlinearProblem(initial,np.ones(size) if variableScale is None else variableScale,
        np.ones(size) if residualScale is None else residualScale,
        np.full(size,-np.inf) if lowerBound is None else lowerBound,
        np.full(size,np.inf) if upperBound is None else upperBound,
        VariablePolicies((),(),()),reference,runtimeFactory,problemIdentity)
