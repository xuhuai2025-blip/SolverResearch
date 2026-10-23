# SolverResearch

面向科学计算的独立 Python 求解库，提供**非线性方程求解**和**刚性 ODE / index-1 DAE 瞬态积分**两类功能。用户定义方程、初值及数值设置，即可计算解并查看收敛信息。

## 选择哪类求解器

| 计算需求 | 求解器 | 输入 | 得到什么 |
|---|---|---|---|
| 找到一组同时满足所有方程的未知量 | `nonlinearSolver` | 残差函数 `F(x)`、未知量初值 | 方程组的根、最终残差和迭代记录 |
| 计算状态随时间的变化 | `dasslSolver` | 微分方程、时间区间和初值 | 状态与导数的时间历程、事件和积分统计 |
| 同时满足微分方程与代数约束 | `dasslSolver` | 隐式残差 `F(t,y,y')` 或质量矩阵形式 | 满足约束的瞬态轨迹及一致初值 |

## 非线性方程求解

用于求解方形实数方程组 `F(x) = 0`，适合稳态平衡、耦合代数方程和离散后形成的非线性方程组。方程数须与未知量数相同。

### 主要功能

- **普通函数直接接入**：提供一个返回残差向量的 Python 函数即可求解。
- **变量与残差缩放**：分别设置尺度，处理不同量级的未知量和方程。
- **边界与定义域控制**：设置变量上下界及用户可行性检查；对已登记变量使用比例限步等策略。
- **稀疏雅可比计算**：逐列有限差分，按已声明的依赖关系计算受影响的残差。
- **局部重算与缓存回退**：显式依赖图支持 AFC/AVC 调度，只重算失效条目；试探结束后恢复接受态和缓存。
- **矩阵分解复用**：复用有效的稀疏分解，并根据线性预测质量决定何时重建。
- **精度恢复**：提供差分间隔调整、病态线性步恢复及达标后的精度复核。
- **独立收敛验收**：最后重新计算完整原方程残差，以用户指定的容差判定收敛。
- **过程控制与诊断**：支持进度回调、取消，以及残差、迭代次数、雅可比构建和分解复用等统计。

普通函数入口保守声明完整依赖。需要利用大型问题的稀疏结构时，可通过 `NonlinearProblem`、`GridProgram` 和 `ScalarRuntime` 显式建立依赖图。

### 最小示例

```python
import numpy as np
from nonlinearSolver import problemFromFunction, solve

# 定义两个未知量、两个方程以及初值。
problem = problemFromFunction(
    lambda x: np.array([x[0]**2 + x[1] - 3, x[0] + x[1]**2 - 3]),
    [1.2, 1.1],
)
state, info = solve(problem, options={'Tolerance': 1e-10, 'Display': 'off'})
print(state, info.Converged, info.FinalResidualInf)
```

详细选项见 [非线性求解接口](src/nonlinearSolver/README.md)。

## 刚性 ODE 与 DAE 瞬态积分

支持以下三种方程形式：

| 方程形式 | 数学表达 | 使用方式 |
|---|---|---|
| 显式 ODE | `y' = f(t,y)` | 提供 `RHS` |
| 质量矩阵形式 | `M(t,y)y' = f(t,y)` | 提供 `Mass` 和 `RHS` |
| 完全隐式残差 | `F(t,y,y') = 0` | 提供 `Residual` |

适合具有快慢时间尺度的刚性系统，以及包含守恒关系、平衡关系等代数约束的 index-1 DAE。

### 主要功能

- **自适应积分**：BDF 1–5 阶，自动调整步长和阶数；可限制最大阶数、初始步长及最大步长。
- **BDF / NDF 选择**：支持纯 BDF 和后端 NDF 修正。
- **一致初值计算**：修正不满足方程的初始状态和导数，并允许固定指定分量。
- **质量矩阵支持**：接收常量、时间相关或状态相关矩阵；奇异质量矩阵直接按隐式残差积分。
- **雅可比与稀疏模式**：接收解析矩阵、回调和稀疏模式；隐式雅可比可仅提供一部分，其余部分由差分补齐。
- **事件检测**：按事件值、穿越方向和终止标记定位事件，可在事件发生时结束积分。
- **自定义输出**：指定输出时间，在已接受步上接收进度回调，并可请求停止。
- **连续查询**：在实际积分区间内任意查询时刻的状态和导数。
- **状态检查**：提供用户定义域检查、显式 ODE 接受点的非负分量检查及步数上限。
- **结果审计**：记录输入与实际初值、最终状态、原方程终态残差，以及步数、阶数、函数求值和矩阵分解统计。

### 最小 DAE 示例

下面同时求解一个微分方程和一个代数约束，解析解为 `y₀ = exp(-t)`、`y₁ = exp(-2t)`。

```python
import numpy as np
from dasslSolver import TransientProblem, TransientSolver

# 固定微分状态初值，允许一致初值计算修正代数分量。
problem = TransientProblem(
    Residual=lambda t, y, yp: np.array([yp[0] + y[0], y[1] - y[0]**2]),
    FixedY0=np.array([True, False]),
    FixedYP0=np.array([False, True]),
)
result = TransientSolver().solve(problem, [0., 1.], [1., .8], [-1., -2.])
print(result.InitialState, result.FinalState, result.FinalResidualInf)
```

详细接口见 [瞬态求解接口](docs/transientApi.md)。该模块使用固定版本的 `scipy-dae` BDF 后端；算法来源及与 MATLAB 实现的差异见 [来源说明](docs/provenance.md)。

## 安装与示例

```powershell
python -m pip install .
```

需要 Python 3.10+、NumPy、SciPy 和 `scipy-dae==0.1.1`。非线性核心附带 Windows x64 原生调度库及 C++17 源码；当前验收环境为 Windows x64。原生库重建方式见 [构建说明](src/nonlinearSolver/native/README.md)。

可直接运行的示例：

- [非线性方程组](examples/nonlinearExample.py)
- [刚性 ODE](examples/stiffExample.py)
- [隐式 DAE 与一致初值](examples/daeExample.py)

## 验证与适用范围

已有 **23 项专项检查通过**，覆盖已知根、尺度与边界、依赖失效、缓存回退、刚性解析解、奇异质量矩阵、一致初值、事件、回调和守恒关系。测试范围与误差记录见 [验证说明](docs/verification.md)。

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

非线性模块面向方形实数方程组；瞬态模块面向刚性 ODE 和 index-1 DAE。非负分量功能检查接受点，不作数值裁剪。数值收敛和终态残差供调用方判断计算质量，领域模型的有效性仍由调用方检查。
