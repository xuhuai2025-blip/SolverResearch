# 算法来源与验证边界

## 稀疏非线性方程

数值核心包含变量/残差尺度、上下界、关键残差限幅、AFC/AVC 依赖调度、逐列差分、事务回退、稀疏分解复用及 float64 精度恢复。复制过程中仅整理公共选项、诊断与说明文字；数值表达式、常量和控制流使用 AST 核对，原生调度库按字节核对。

代码使用 NumPy、SciPy 的 SuperLU/LSMR 和独立 C++ 调度器，不是其他商业软件的原生求解库。当前核心的数值规则不自动扩展为最小二乘、优化、DAE 或任意维度问题。

## 瞬态接口

Python 版转写自本仓库先前的 [DASSLTransient/Solver/+dassl/TransientSolver.m](https://github.com/xuhuai2025-blip/SolverResearch/blob/77b378a16e5d20fb2e67d7c3484d7e6e28dfb4f5/DASSLTransient/Solver/%2Bdassl/TransientSolver.m) 的通用接口设计和测试问题。原实现调用 MATLAB `ode15s`、`ode15i` 和 `decic`，没有独立的 DASSL/DASPK 内核。

Python 版采用 [SolveDAE 的 Python BDF/DAE 实现](https://github.com/SolveDAE/solve_dae)，通过发布包 `scipy-dae==0.1.1` 固定后端版本，使用其 BDF 核心、事件寻根、一致初值和连续输出；本仓库维护问题形式转换、合同检查、回调、阶数和输出组织。第三方后端按其自身 BSD-3-Clause 许可分发，本仓库不将其算法声称为自写 DASSL 内核。

`BDF=True` 明确关闭 NDF 修正，使用后端的纯 BDF 系数；`BDF=False` 使用该后端的 NDF 系数。最高阶数限制为 1–5。单状态插值的标量形状通过包装恢复为一维状态向量，插值多项式不变。

保留的功能为显式 ODE、常量/时间相关/状态相关质量矩阵、残差 DAE、固定初值分量、解析/稀疏/部分雅可比、事件、输出回调、连续查询和终态原方程审计。原 MATLAB 专属后端名称和无效选项不作为兼容分支保留，差异详见接口说明。

两个后端的步长、阶数选择和误差估计不同，本次以相同方程的解析解和守恒关系核验，不声称 Python 与 MATLAB 数值轨迹逐步相同，也不声称等同原版 DASSL 或 SUNDIALS IDA。
