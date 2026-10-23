"""用户方程与数值核心的接口：固定问题定义和每次求解独立的运行时。"""
from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np


# 维护数值状态、依赖或计算接口。
class ReferenceEvaluator(Protocol):
    # 维护数值状态、依赖或计算接口。
    def evaluateTrial(self, state): ...
    # 维护数值状态、依赖或计算接口。
    def evaluateResidual(self, state): ...


from nonlinearSolver.errors import EvaluationDomainError


# 维护数值状态、依赖或计算接口。
@dataclass(frozen=True)
class NonlinearProblem:
    """只携带数值合同，不包含领域对象或界面字段。

    数组在构造时复制并只读。领域适配器负责创建求值器、运行时及参数身份检查。
    每次 factory 调用必须返回全新的缓存/状态，不能跨求解共享可变图。
    """
    initialState: np.ndarray
    variableScale: np.ndarray
    residualScale: np.ndarray
    lowerBound: np.ndarray
    upperBound: np.ndarray
    policies: object
    reference: ReferenceEvaluator
    runtimeFactory: Callable = field(repr=False)
    identity: Callable = field(repr=False)
    isDomainError: Callable = field(default=lambda error:isinstance(error,EvaluationDomainError),repr=False)

    # 维护数值状态、依赖或计算接口。
    def __post_init__(self):
        size=np.size(self.initialState)
        for name in ('initialState','variableScale','residualScale','lowerBound','upperBound'):
            value=np.asarray(getattr(self,name),float).copy()
            if value.shape!=(size,):raise ValueError('numerical:ProblemShape: '+name)
            value.flags.writeable=False
            object.__setattr__(self,name,value)
        # 尺度必须为有限正数，避免无穷尺度把非零残差压成零，或让 NaN 漂入雅可比。
        for name in ('variableScale','residualScale'):
            value=getattr(self,name)
            if not np.all(np.isfinite(value)&(value>0)):
                raise ValueError('numerical:ProblemScales: '+name+' 必须为有限正数')
        if np.any(self.lowerBound>self.upperBound):raise ValueError('numerical:ProblemBounds')

    # 维护数值状态、依赖或计算接口。
    def createRuntime(self):
        """每次求解创建独立的当前运行时，不提供旧线性化模式。"""
        return self.runtimeFactory()
