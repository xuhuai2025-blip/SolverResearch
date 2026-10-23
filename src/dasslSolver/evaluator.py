"""把三种方程形式转换为 F(t,y,yp)=0，并组织雅可比合同。"""
import numpy as np
from scipy import sparse
from dasslSolver.problem import matrix, vector


def difference(function, state):
    """以独立逐列前向差分补齐用户未提供的雅可比部分。"""
    state = np.asarray(state,float)
    base = function(state)
    result = np.empty((state.size,state.size))
    for column in range(state.size):
        changed = state.copy()
        step = np.sqrt(np.finfo(float).eps)*max(1.,abs(state[column]))
        changed[column] += step
        result[:,column] = (function(changed)-base)/(changed[column]-state[column])
    return result


class Evaluator:
    """对回调返回值、数值域及矩阵形状作一致检查。"""
    def __init__(self, problem, size):
        """绑定问题，不推断领域方程或初值。"""
        self.problem = problem
        self.size = size

    def validate(self, time, state):
        """用户定义域检查在每次残差求值前执行。"""
        vector(state,'状态',self.size)
        callback = self.problem.ValidateState
        if callback is not None:
            accepted = callback(time,state.copy())
            if accepted is not None and not np.all(accepted):
                raise ValueError('dassl:StateDomain: 用户定义域检查拒绝该状态')

    def mass(self, time, state):
        """支持常矩阵、M(t) 和 M(t,y)，不求逆奇异质量矩阵。"""
        value = self.problem.Mass
        if value is None:
            return np.eye(self.size)
        if callable(value):
            value = value(time) if self.problem.MassStateDependence == 'none' else value(time,state.copy())
        return matrix(value,self.size,'Mass')

    def rhs(self, time, state):
        """计算并检查显式右端。"""
        self.validate(time,state)
        return vector(self.problem.RHS(time,state.copy()),'RHS',self.size)

    def residual(self, time, state, derivative):
        """奇异质量矩阵直接组成残差，不按伪逆转换为 ODE。"""
        self.validate(time,state)
        derivative = vector(derivative,'导数',self.size)
        if self.problem.form == 'residual':
            return vector(self.problem.Residual(time,state.copy(),derivative.copy()),'Residual',self.size)
        rhs = self.rhs(time,state)
        return self.mass(time,state)@derivative-rhs

    def jacobian(self, time, state, derivative):
        """返回 dF/dy 和 dF/dyp；允许隐式接口只提供其中一部分。"""
        problem = self.problem
        supplied = problem.Jacobian
        if problem.form == 'residual':
            values = supplied(time,state.copy(),derivative.copy()) if callable(supplied) else supplied
            if not isinstance(values,(tuple,list)) or len(values) != 2:
                raise ValueError('dassl:Jacobian: 隐式雅可比必须返回两个矩阵或由两个矩阵组成')
            jy, jyp = values
            if jy is None:
                jy = difference(lambda y:self.residual(time,y,derivative),state)
            if jyp is None:
                jyp = difference(lambda yp:self.residual(time,state,yp),derivative)
        else:
            # 右端雅可比不包括状态相关质量矩阵的导数，需要另行补齐。
            value = supplied(time,state.copy()) if callable(supplied) else supplied
            jy = -matrix(value,self.size,'RHS Jacobian')
            if callable(problem.Mass) and problem.MassStateDependence != 'none':
                jy = jy+difference(lambda y:self.mass(time,y)@derivative,state)
            jyp = self.mass(time,state)
        return matrix(jy,self.size,'dF/dy'), matrix(jyp,self.size,'dF/dyp')

    def denseJacobian(self, time, state, derivative):
        """一致初值使用稠密 QR；转换稀疏解析矩阵以匹配后端合同。"""
        values = self.jacobian(time,state,derivative)
        return tuple(value.toarray() if sparse.issparse(value) else value for value in values)

    def patterns(self, time, state):
        """将用户稀疏模式转换为残差的两个偏导模式。"""
        problem = self.problem
        pattern = problem.JacobianPattern
        if pattern is None:
            return None
        if problem.form == 'residual':
            if not isinstance(pattern,(tuple,list)) or len(pattern)!=2:
                raise ValueError('dassl:JacobianPattern: 隐式稀疏模式必须是两个矩阵')
            return tuple(np.ones((self.size,self.size)) if value is None else matrix(value,self.size,'JacobianPattern') for value in pattern)
        jy = matrix(pattern,self.size,'JacobianPattern')
        if callable(problem.Mass) and problem.MassStateDependence != 'none':
            additional = np.ones((self.size,self.size)) if problem.MassVectorPattern is None else matrix(problem.MassVectorPattern,self.size,'MassVectorPattern')
            jy = sparse.csc_matrix(jy)+sparse.csc_matrix(additional)
        # 时间相关质量矩阵保守声明完整模式，不能仅用初始零元素排除依赖。
        jyp = np.ones((self.size,self.size)) if callable(problem.Mass) else self.mass(time,state)
        return jy,jyp
