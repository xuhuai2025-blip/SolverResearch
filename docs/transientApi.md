# 瞬态求解接口

`TransientSolver(options).solve(problem,timeSpan,initialState,initialDerivative=None)` 和函数式 `solve(problem,timeSpan,initialState,options=None,initialDerivative=None)` 使用同一实现。

## 方程形式

- `RHS(t,y)`：显式方程 `y'=f(t,y)`。
- `RHS` 与 `Mass`：`M(t,y)y'=f(t,y)`。质量矩阵可为固定矩阵、`M(t)`（`MassStateDependence='none'`）或 `M(t,y)`。
- `Residual(t,y,yp)`：完整的隐式方程 `F(t,y,yp)=0`；此时不能再设置 `Mass`。

状态为一维有限实数向量，时间有限且严格递增。支持方形、微分指数为 1 的 DAE；不承诺高指数系统。

## 设置

| 设置 | 默认值 | 说明 |
|---|---|---|
| `RelTol` / `AbsTol` | `1e-6` / `1e-8` | 正相对容差、正标量或逐分量绝对容差 |
| `MaxStep` / `InitialStep` | 自动 | 最大步长、初始步长 |
| `MaxOrder` | 5 | 最高 BDF 阶数，范围 1–5 |
| `BDF` | True | 纯 BDF；False 使用后端 NDF 修正 |
| `OutputTimes` | timeSpan | 请求的输出时间，位于积分区间内，可重复 |
| `ComputeConsistentInitialConditions` | True | 残差或质量矩阵问题的一致初值计算 |
| `Stats` | False | 打印简要积分统计 |
| `MaxSteps` | 100000 | 已接受步数上限 |

`FixedY0`、`FixedYP0` 为与状态等长的布尔掩码，True 表示一致初值计算时保持该分量。显式 ODE 固定初始状态，导数由 RHS 给出。关闭一致初值计算时，输入起点仍须通过残差检查。

## 雅可比、事件和回调

隐式 `Jacobian` 为两个矩阵 `(dF/dy,dF/dyp)` 或返回这一对矩阵的回调 `(t,y,yp)`；可以用 None 缺省一部分，由差分补齐。`JacobianPattern` 同样为两个模式。RHS 问题使用单个 `df/dy` 矩阵或 `(t,y)` 回调，以及一个 RHS 稀疏模式。

状态相关质量矩阵的导数会补入残差雅可比；`MassVectorPattern` 可声明相应稀疏模式，不提供时保守采用完整依赖。`MassSingular` 为 `yes/no/maybe` 结构标记，实际积分始终按残差处理质量矩阵，不据此将奇异系统改写为显式 ODE。

`Events` 返回 `(value,terminal,direction)` 三元组。ODE/质量矩阵回调为 `(t,y)`；隐式回调为 `(t,y,yp)`。每个事件支持终止标记和穿越方向。

`OutputFcn(t,y,flag)` 的 flag 为 `init/step/done`，在已接受步上收到状态副本；step 返回 True 可停止。init 只用于通知，不能请求停止。`ValidateState(t,y)` 在每次残差调用前检查定义域。`NonNegative` 使用从 0 起始的整数分量索引，仅检查显式 ODE 接受点，不投影或裁剪负值。

所有回调的返回值始终检查形状和有限性。原 MATLAB 的 `Backend='ode15s/ode15i'`、`Refine`、`RuntimeChecks` 和 `ThrowOnNonfiniteResidual` 不作为无效兼容选项保留；未知选项明确报错。

## 结果

结果提供 `Time`、`State`、`Derivative`、输入和实际初值、终态、`FinalResidualInf`、后端名称、步数、阶数、函数/雅可比/分解计数及分阶段耗时。`Evaluate(t)` 返回状态和导数，拒绝实际积分区间之外的查询。

`EndReason` 为 `final_time/event/output_callback`。最终时间总会加入输出；终止事件后的请求时间会剔除。接受步记录保留后端工作步，事件可能位于最后一个接受步的内部。

终态残差由原始方程独立重新计算，是连续输出的诊断值，不是后端内部 Newton 容差的证明。
