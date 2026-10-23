"""刚性 ODE、质量矩阵和 index-1 DAE 的 Python 积分接口。"""
from dasslSolver.problem import TransientProblem
from dasslSolver.solver import TransientSolver, IntegrationResult, solve
from dasslSolver.validation import validateProblem, debug

__all__=['TransientProblem','TransientSolver','IntegrationResult','solve','validateProblem','debug']
