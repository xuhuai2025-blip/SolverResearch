"""非线性控制策略。用户方程、关键行与变量合法性分别管理。"""
from dataclasses import dataclass
import numpy as np
from nonlinearSolver.dependencyGraph import indices

LEVEL_FRACTIONS=(.0159,.0251,.0398,.0631,.1,.251,.398,.631,1.)


# 维护数值状态、依赖或计算接口。
def initialFDInterval(options):
    """2*sqrt(2000*Roundoff)，Roundoff 按 float64 计算。

    显式正值用于诊断；0 使用统一自动值。
    """
    step=float(options.get('FiniteDifferenceStep',0))
    if not np.isfinite(step) or step<0:raise ValueError('nonlinear:FiniteDifferenceStep')
    return step if step else 2*np.sqrt(2000*np.finfo(float).eps/2)


# 维护数值状态、依赖或计算接口。
@dataclass(frozen=True)
class VariablePolicies:
    keyRows: tuple
    positiveRows: tuple
    divergingRows: tuple


    # 维护数值状态、依赖或计算接口。
    def legalStep(self,current,trial,positiveLimit=True):
        if positiveLimit and self.positiveRows:
            rows=np.asarray(self.positiveRows,int)
            if np.any((trial[rows]<=current[rows]/1.4)|(trial[rows]>=1.4*current[rows])):
                return False,'positive_ratio'
        if self.divergingRows:
            rows=np.asarray(self.divergingRows,int)
            if np.any(abs(trial[rows]-current[rows])>=np.maximum(abs(current[rows]),1.)):
                return False,'diverging_step'
        return True,''


# 维护数值状态、依赖或计算接口。
class TargetResidual:
    """关键残差限幅。运行峰值为已声明近似，不冒充其他实现 有效栈生命周期。"""
    # 初始化数值对象。
    def __init__(self,rows,level):
        if type(level) is not int or not 0<=level<=8:raise ValueError('nonlinear:StepLevel')
        self.rows=np.asarray(rows,int);self.fraction=LEVEL_FRACTIONS[level]
        self.peak=np.ones(len(self.rows))

    # 维护数值状态、依赖或计算接口。
    def calculate(self,residual):
        result=residual.copy();self.peak=np.maximum(self.peak,abs(residual[self.rows]))
        cap=self.fraction*self.peak
        result[self.rows]=np.clip(residual[self.rows],-cap,cap)
        return result
