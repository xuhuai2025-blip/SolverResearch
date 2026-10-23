# 原生依赖调度器

维护标量依赖、有效位、批量计算及 journal 回退。ABI 为 2；不包含用户方程。

安装后运行 `python -m nonlinearSolver.native.build` 可重建当前系统的库。支持 Windows x64/MSVC C++17 和 macOS arm64/Apple Clang。编译保留严格浮点设置，不启用 fast-math。

加载器检查平台、架构、ABI 和结构布局。文件摘要及架构会记录在结果统计中。试探回退不会撤销已完成工作的统计计数。
