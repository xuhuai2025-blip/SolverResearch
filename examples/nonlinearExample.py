"""求解具有已知根的普通耦合代数方程。"""
import json
import numpy as np
from nonlinearSolver import problemFromFunction, solve

# 定义残差及普通初值。
problem=problemFromFunction(lambda x:np.array([x[0]**2+x[1]-3,x[0]+x[1]**2-3]),[1.2,1.1])
state,info=solve(problem,options={'Tolerance':1e-10,'Display':'off'})
print(json.dumps({'根':state.tolist(),'收敛':info.Converged,'完整残差无穷范数':info.FinalResidualInf},ensure_ascii=False,indent=2))
