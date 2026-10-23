"""求解微分方程和代数约束，并校正不一致初值。"""
import json
import numpy as np
from dasslSolver import TransientProblem, solve

# 固定微分初始状态，由一致初值计算修正代数分量。
problem=TransientProblem(Residual=lambda t,y,yp:np.array([yp[0]+y[0],y[1]-y[0]**2]),
    FixedY0=np.array([True,False]),FixedYP0=np.array([False,True]))
result=solve(problem,np.linspace(0.,1.,101),[1.,.8],{'RelTol':1e-7,'AbsTol':1e-9},[-1.,-2.])
exact=np.array([np.exp(-result.Time),np.exp(-2*result.Time)])
print(json.dumps({'输入初值':result.InputInitialState.tolist(),'一致初值':result.InitialState.tolist(),
    '最大状态误差':float(np.max(abs(result.State-exact))),
    '终态残差无穷范数':result.FinalResidualInf},ensure_ascii=False,indent=2))
