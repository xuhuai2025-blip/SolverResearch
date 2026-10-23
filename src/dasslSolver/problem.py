"""刚性 ODE、质量矩阵与隐式残差问题的输入合同。"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse


def vector(value, name, size=None):
    """接受标量或一维有限实数向量，不将矩阵静默展平。"""
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError('dassl:Vector: '+name+'必须为有限实数标量或一维向量')
    if array.size == 0 or size is not None and array.size != size:
        raise ValueError('dassl:VectorSize: '+name+'的长度必须与状态一致且非空')
    return np.asarray(array, float).copy()


def matrix(value, size, name):
    """检查解析雅可比、质量矩阵和稀疏模式的形状。"""
    if sparse.issparse(value):
        result = sparse.csc_matrix(value, dtype=float)
        finite = np.isfinite(result.data).all()
    else:
        result = np.asarray(value)
        if result.ndim == 0 and size == 1:
            result = result.reshape(1,1)
        if result.dtype.kind not in 'biuf':
            raise ValueError('dassl:Matrix: '+name+'必须为实数矩阵')
        finite = np.isfinite(result).all()
        result = np.asarray(result, float)
    if result.shape != (size,size) or not finite:
        raise ValueError('dassl:MatrixShape: '+name+'必须是与状态维数相同的有限方阵')
    return result


@dataclass
class TransientProblem:
    """用户方程及可选的导数、事件、输出和初值约束。"""
    RHS: object = None
    Residual: object = None
    Mass: object = None
    Jacobian: object = None
    JacobianPattern: object = None
    FixedY0: object = None
    FixedYP0: object = None
    MassStateDependence: str = 'weak'
    MassVectorPattern: object = None
    MassSingular: str = 'maybe'
    Events: object = None
    OutputFcn: object = None
    ValidateState: object = None
    NonNegative: object = None

    @property
    def form(self):
        """优先选择隐式残差，其次质量矩阵，最后显式 ODE。"""
        if self.Residual is not None:
            return 'residual'
        return 'mass' if self.Mass is not None else 'explicit'


def asProblem(problem):
    """只接受明确的问题对象或字段字典，拒绝未知字段。"""
    if isinstance(problem,dict):
        try:
            problem = TransientProblem(**problem)
        except TypeError as error:
            raise ValueError('dassl:ProblemFields: 问题包含未知字段') from error
    if not isinstance(problem,TransientProblem):
        raise TypeError('dassl:Problem: 请提供 TransientProblem 或问题字段字典')
    selected = problem.Residual if problem.form == 'residual' else problem.RHS
    if not callable(selected):
        raise ValueError('dassl:Problem: 必须提供可调用的 Residual 或 RHS')
    if problem.form == 'residual' and problem.Mass is not None:
        raise ValueError('dassl:Problem: 隐式 Residual 已定义完整方程，不能再设置 Mass')
    if problem.MassStateDependence not in ('none','weak','strong'):
        raise ValueError('dassl:MassStateDependence: 请选择 none、weak 或 strong')
    if problem.MassSingular not in ('yes','no','maybe'):
        raise ValueError('dassl:MassSingular: 请选择 yes、no 或 maybe')
    for name in ('Events','OutputFcn','ValidateState'):
        if getattr(problem,name) is not None and not callable(getattr(problem,name)):
            raise ValueError('dassl:Callback: '+name+'必须为可调用对象')
    return problem
