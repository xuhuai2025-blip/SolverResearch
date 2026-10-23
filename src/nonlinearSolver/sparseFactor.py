"""稀疏矩阵与数值分解生命周期，和非线性接受策略解耦。"""
import time
import numpy as np
from scipy.sparse.linalg import splu, LinearOperator, onenormest, lsmr


# 维护数值状态、依赖或计算接口。
class SparseFactor:
    # 建立独立分解生命周期及精度诊断记录。
    def __init__(self):
        self.factor=None
        self.matrix=None
        self.factorizations=0
        self.solves=0
        self.factorSeconds=0.
        self.solveSeconds=0.
        self.conditionEstimate=None
        self.regularizedSteps=[]

    # 使旧分解失效，下一轮必须重新线性化。
    def invalidate(self):
        self.factor=None

    # 使用当前矩阵建立稀疏直接分解，并清除旧矩阵的条件估计。
    def factorize(self,matrix):
        started=time.perf_counter()
        # 使用 SuperLU 的 COLAMD 排序建立稀疏直接分解。
        self.factor=splu(matrix.tocsc(),permc_spec='COLAMD')
        self.matrix=matrix
        self.factorSeconds+=time.perf_counter()-started
        self.factorizations+=1
        self.conditionEstimate=None

    # 求解原矩阵或转置矩阵，统计真实的三角回代次数。
    def solve(self,rhs,transpose=False):
        if self.factor is None:raise RuntimeError('nonlinear:MissingFactor: 当前没有有效的矩阵分解')
        started=time.perf_counter();step=self.factor.solve(rhs,trans='T' if transpose else 'N')
        self.solveSeconds+=time.perf_counter()-started;self.solves+=1
        if not np.isfinite(step).all():raise ValueError('nonlinear:NonfiniteLinearStep: 线性步包含非有限数值')
        return step

    # 只在受限步上估计病态程度，避免把正常非线性曲率误判为浮点精度丢失。
    def precisionLimited(self):
        if self.conditionEstimate is None:
            inverse=LinearOperator(self.matrix.shape,matvec=self.solve,
                rmatvec=lambda value:self.solve(value,True),dtype=float)
            self.conditionEstimate=float(abs(self.matrix).sum(axis=0).max())*onenormest(inverse,t=1)
        return self.conditionEstimate*np.finfo(float).eps>=1

    # float64 无法可靠分辨弱方向时求最小改变量；阻尼仅作用于线性步，不改原方程。
    def regularizedSolve(self,rhs):
        started=time.perf_counter();roundoff=np.finfo(float).eps
        damping=np.sqrt(roundoff)*float(np.linalg.norm(self.matrix.data))
        result=lsmr(self.matrix,rhs,damp=damping,atol=roundoff**.75,
            btol=roundoff**.75,maxiter=4*self.matrix.shape[0])
        self.solveSeconds+=time.perf_counter()-started
        self.regularizedSteps.append(dict(Damping=damping,StopCode=int(result[1]),
            Iterations=int(result[2]),ConditionEstimate=self.conditionEstimate,
            LinearResidualNorm=float(result[3]),NormalResidualNorm=float(result[4])))
        if not np.isfinite(result[0]).all():raise ValueError('numerical:RegularizedStep: 病态线性步包含非有限数值')
        return result[0]
