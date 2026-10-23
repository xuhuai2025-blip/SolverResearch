# SolverResearch

独立 Python 数值求解库，包含两类求解器。用户提供方程、初值、尺度或时间区间即可调用。

| 包 | 问题 | 主要能力 |
|---|---|---|
| `nonlinearSolver` | 方形实数非线性方程组 `F(x)=0` | 稀疏差分、AFC/AVC 依赖调度、边界限步、分解复用、精度恢复及独立残差验收 |
| `dasslSolver` | 刚性 ODE 与 index-1 DAE | BDF 1–5 阶、自适应步长、一致初值、质量矩阵、隐式残差、雅可比、事件及连续输出 |

## 安装

```powershell
python -m pip install .
```

需要 Python 3.10+。数值依赖为 NumPy、SciPy 和固定版本 `scipy-dae==0.1.1`。当前验收环境为 Windows x64；非线性核心附带 Windows x64 原生调度库及 C++17 源码。`scipy-dae` 自身的安装依赖包含 Matplotlib，示例不需要绘图。

非线性原生库的构建入口为 `python -m nonlinearSolver.native.build`；现有构建脚本支持 Windows x64、macOS arm64，本次没有进行其他系统实机验收。瞬态包为 Python 实现，使用第三方数值后端。

## 非线性方程

```python
import numpy as np
from nonlinearSolver import problemFromFunction, solve

# 用户只需定义普通残差函数，不需要建立任何领域模型。
problem = problemFromFunction(
    lambda x: np.array([x[0]**2+x[1]-3, x[0]+x[1]**2-3]),
    [1.2, 1.1],
)
state, info = solve(problem, options={'Tolerance':1e-10, 'Display':'off'})
print(state, info.Converged, info.FinalResidualInf)
```

普通函数保守声明完整依赖；需要稀疏局部重算时，可通过 `NonlinearProblem`、`GridProgram`、`ScalarRuntime` 显式登记依赖。每次求解独享运行时、缓存和回退事务。最终收敛以独立完整残差判断。

## 刚性 ODE

```python
import numpy as np
from dasslSolver import TransientProblem, TransientSolver

# 解析解为 cos(t) 的刚性方程。
problem = TransientProblem(RHS=lambda t,y: -1000*(y-np.cos(t))-np.sin(t))
result = TransientSolver({'RelTol':1e-7,'AbsTol':1e-9}).solve(
    problem, np.linspace(0.,1.,101), [1.])
print(result.FinalState, result.FinalResidualInf)
```

## 隐式 DAE

```python
import numpy as np
from dasslSolver import TransientProblem, TransientSolver

# 微分方程与代数约束：y0'= -y0，y1=y0²。
problem = TransientProblem(
    Residual=lambda t,y,yp: np.array([yp[0]+y[0], y[1]-y[0]**2]),
    FixedY0=np.array([True,False]),
    FixedYP0=np.array([False,True]),
)
result = TransientSolver().solve(problem,[0.,1.],[1.,.8],[-1.,-2.])
print(result.InitialState, result.FinalState)
```

质量矩阵问题可通过 `Mass` 和 `RHS` 输入，不需要显式求逆质量矩阵；奇异质量矩阵以隐式残差积分。

## 验证和说明

```powershell
python -m unittest discover -s tests -p "test*.py" -v
python examples/nonlinearExample.py
python examples/stiffExample.py
python examples/daeExample.py
```

详细接口见 [非线性模块](src/nonlinearSolver/README.md)、[瞬态接口](docs/transientApi.md)，算法来源和后端差异见 [来源与边界](docs/provenance.md)，验证结果见 [验证记录](docs/verification.md)。
