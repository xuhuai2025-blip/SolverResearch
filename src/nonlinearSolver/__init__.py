"""独立的稀疏非线性方程组模块；仅提供当前 AFC/AVC 求解算法。"""
from nonlinearSolver.problem import NonlinearProblem
from nonlinearSolver.functionProblem import problemFromFunction
from nonlinearSolver.solver import NonlinearSolver, solve

__all__=['NonlinearProblem','NonlinearSolver','problemFromFunction','solve']
