"""求解解析解为 cos(t) 的刚性 ODE。"""
import json
import numpy as np
from dasslSolver import TransientProblem, solve

# 使用解析轨迹计算全输出区间误差。
problem=TransientProblem(RHS=lambda t,y:-1000*(y-np.cos(t))-np.sin(t))
result=solve(problem,np.linspace(0.,1.,101),[1.],{'RelTol':1e-7,'AbsTol':1e-9})
print(json.dumps({'最大状态误差':float(np.max(abs(result.State[0]-np.cos(result.Time)))),
    '终态残差无穷范数':result.FinalResidualInf,'内部网格点':result.InternalMeshPointCount},ensure_ascii=False,indent=2))
