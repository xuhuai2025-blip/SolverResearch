# 稀疏非线性方程求解

求解实数方形方程组 `F(x)=0`。核心包含 AFC/AVC 逐标量依赖、失效传播、局部重算、journal 回退、逐列有限差分、稀疏直接分解复用及 float64 病态线性步恢复。

公开入口为 `problemFromFunction`、`NonlinearProblem`、`NonlinearSolver` 和 `solve`。普通回调返回与未知量等长的有限实数向量；维度不能在求解中变化。

`problemFromFunction(residual,initialState,variableScale=None,residualScale=None,lowerBound=None,upperBound=None,isFeasible=None,identity=None)` 支持尺度、边界和用户定义域。每次工厂调用生成独立的运行时和缓存。外部参数会变化时，提供 `identity` 回调，以便在雅可比构造前后检查问题是否改变。

| 求解选项 | 含义 |
|---|---|
| `Tolerance` | 缩放后完整残差的无穷范数阈值 |
| `MaxIterations` | 最大迭代次数 |
| `FiniteDifferenceStep` | 差分间隔，0 表示按 float64 舍入尺度自动设置 |
| `StepLevel` | 0–8，关键残差目标限幅级别 |
| `PositiveRatioLimit` | 是否启用已登记正变量的比例限步 |
| `Display` | `off` 关闭逐轮输出 |
| `CancellationCheck` | 返回真时取消的回调 |
| `ProgressFcn` | 每轮进度记录回调 |

通过 `GridProgram` 显式建图时，展示标签不决定算子类型；自定义函数按真实回调执行。最终收敛必须通过独立、无缓存的完整残差复核。LSMR 仅作为病态线性步恢复，不是另一套主求解器。

此模块需要匹配架构的 ABI 2 原生库。Windows x64 使用 `native/nonlinear_runtime.dll`；macOS arm64 需在目标机器构建 `native/libnonlinear_runtime.dylib`。可通过 `NONLINEAR_SOLVER_LIBRARY` 选择本机匹配库。

需要手工重建时，在已安装的环境执行 `python -m nonlinearSolver.native.build`。Windows 使用 MSVC C++17 `/fp:strict`；macOS 使用 Apple Clang、关闭 fast-math 和浮点融合。库被进程加载后，须关闭持有它的进程再替换文件。
