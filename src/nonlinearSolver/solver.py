"""稀疏非线性状态机：AFC/AVC、目标残差、复用、回退。

float64、SuperLU 和显式块依赖是本实现选择。病态线性步恢复是数值后端适配。
该后端不采用着色或 GMRES；终态始终独立验算完整原方程。
"""
import time
import numpy as np
from nonlinearSolver.values import Struct
from nonlinearSolver.jacobian import FiniteDifferenceWorker, buildJacobian, updateIntervals
from nonlinearSolver.policies import TargetResidual, initialFDInterval
from nonlinearSolver.sparseFactor import SparseFactor


# 维护数值状态、依赖或计算接口。
class NonlinearSolver:
    # 绑定独立数值问题，建立当前求解的分解与统计容器。
    def __init__(self,problem,options):
        supported={'MaxIterations','Tolerance','FiniteDifferenceStep','Display','StepLevel',
                   'PositiveRatioLimit','CancellationCheck','ProgressFcn'}
        unknown=set(options)-supported
        if unknown:raise ValueError('nonlinear:Options: 不支持的求解选项：'+', '.join(sorted(unknown)))
        # 在创建运行时和求值用户方程前拒绝无效容差，不能借无穷阈值报告收敛。
        tolerance=float(options.get('Tolerance',1e-8))
        if not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('nonlinear:SolverOptions: 容差必须为有限正数')
        self.problem=problem;self.options=options
        self.worker=FiniteDifferenceWorker(problem)
        self.reference=problem.reference
        self.policies=problem.policies
        self.factor=SparseFactor()
        self.modelFingerprint=problem.identity()
        self.counts=Struct(FunctionEvaluations=0,JacobianBuilds=0,FactorReuses=0,
            FDIntervalUpdates=0,FDReversals=0,FDReductions=0,TrialRejects=0,
            JacobianSeconds=0.,FDUpdateSeconds=0.,LineSearchSeconds=0.,
            PrecisionRecoverySeconds=0.)

    # 在运算边界响应用户取消。
    def checkCancel(self):
        callback=self.options.get('CancellationCheck')
        if callback and callback():raise InterruptedError('nonlinear:Cancelled: 用户已取消求解')

    # 按变量和残差尺度缩放数值问题并求解。
    def solve(self,initial):
        self.vs=np.asarray(self.problem.variableScale,float)
        self.rs=np.asarray(self.problem.residualScale,float)
        if np.any(self.vs<=0) or np.any(self.rs<=0):raise ValueError('nonlinear:Scales: 变量与残差尺度必须为正数')
        state,info=self._iterate(np.asarray(initial,float)/self.vs)
        info.ExecutionBackend='single_process_native_grid'
        info.FunctionEvaluationsComplete=True
        return state*self.vs,info

    # 统一检查变量边界和用户可行性，失败试探不提交。
    def _evaluate(self,state,accepted):
        self.checkCancel();physical=state*self.vs
        c=self.problem
        if not np.all(np.isfinite(physical)&(physical>=c.lowerBound)&(physical<=c.upperBound)):
            self.counts.TrialRejects+=1;return None,False
        valid,_=self.policies.legalStep(accepted*self.vs,physical,self.options.get('PositiveRatioLimit',True))
        if not valid:self.counts.TrialRejects+=1;return None,False
        self.counts.FunctionEvaluations+=1
        try:residual,valid=self.reference.evaluateTrial(physical)
        except Exception as error:
            if self.problem.isDomainError(error):self.counts.TrialRejects+=1;return None,False
            raise
        if not valid:self.counts.TrialRejects+=1;return None,False
        residual=np.asarray(residual,float)
        if residual.shape!=state.shape or not np.isfinite(residual).all():raise ValueError('nonlinear:ResidualContract: 残差形状必须与状态一致且数值有限')
        return residual/self.rs,True

    # 按潜在依赖逐列构建雅可比，再建立新的稀疏分解。
    def _build(self,state,intervals):
        self.checkCancel();started=time.perf_counter()
        if self.problem.identity()!=self.modelFingerprint:
            raise ValueError('nonlinear:ModelChangedBeforeJacobian: 构建雅可比前模型输入已变化')
        matrix,parts=buildJacobian(self.worker,state,intervals,self.vs,self.rs,
            chunk=32,cancel=self.options.get('CancellationCheck'))
        if self.problem.identity()!=self.modelFingerprint:
            raise ValueError('nonlinear:ModelChangedDuringJacobian: 构建雅可比期间模型输入发生变化')
        seconds=time.perf_counter()-started
        self.counts.JacobianSeconds+=seconds;self.counts.JacobianBuilds+=1
        self.counts.FDReversals+=sum(p['reversals'] for p in parts)
        self.counts.FDReductions+=sum(p['reductions'] for p in parts)
        self.counts.FunctionEvaluations+=len(state)
        self.counts.ObservedJacobianNonzeros=matrix.nnz
        workers=self.counts.setdefault('WorkerRuntimeStats',{})
        for part in parts:
            workers[str(part['pid'])]=dict(runtime=part['runtimeStats'],properties=part['propertyStats'])
        self.factor.factorize(matrix)
        return seconds

    # 维护数值状态、依赖或计算接口。
    def _trial(self,state,step,alpha=1.):
        """变量合法性限步；不在这里改变残差下降和新/旧分解接受规则。"""
        trials=0
        while alpha>0:
            candidate=state+alpha*step
            # 逐次减半检查变量；以数值已无法改变状态作为终止边界。
            if np.array_equal(candidate,state):break
            new,valid=self._evaluate(candidate,state);trials+=1
            if valid:return candidate,new,alpha,trials
            alpha*=.5
        return None,None,alpha,trials

    # float64 在首次达标后重新线性化一次，减少弱耦合输出对剩余状态误差的放大。
    def _verifyConverged(self,state,residual,intervals,tolerance):
        verification=Struct(Applied=False,BeforeResidualInf=float(np.max(abs(residual),initial=0)))
        try:
            self._build(state,intervals)
            correction=self.factor.solve(-residual)
        except RuntimeError as error:
            # 已达标状态可能存在不可辨识的自由方向；仅识别明确的奇异分解错误。
            if 'singular' not in str(error).lower():raise
            verification.Reason='精度复核矩阵奇异，保留已验收状态'
            return state,residual,verification
        candidate=state+correction;new,valid=self._evaluate(candidate,state)
        verification.CorrectionInf=float(np.max(abs(correction),initial=0))
        if valid:
            verification.CandidateResidualInf=float(np.max(abs(new),initial=0))
            # 仅接受完整且仍通过原阈值的校正，不以缩步或放宽阈值制造复核通过。
            if verification.CandidateResidualInf<=tolerance:
                verification.Applied=True
                return candidate,new,verification
        verification.Reason='完整校正未通过原验收或变量约束，保留已验收状态'
        return state,residual,verification

    # 组织残差限幅、接受回退、分解复用及数值后端精度恢复。
    def _iterate(self,state):
        o=self.options;started=time.perf_counter();history=[];tol=float(o.get('Tolerance',1e-8))
        maximum=int(o.get('MaxIterations',80));level=int(o.get('StepLevel',4))
        targetPolicy=TargetResidual(self.policies.keyRows,level)
        stepSize=initialFDInterval(o)
        if not np.isfinite(stepSize) or stepSize<=0 or tol<=0 or maximum<1:raise ValueError('nonlinear:SolverOptions: 差分间隔、容差及迭代次数设置无效')
        intervals=np.full(len(state),stepSize)
        residual,valid=self._evaluate(state,state)
        if not valid:raise ValueError('nonlinear:InitialState: 初始状态不满足变量或用户定义域约束')
        rms=lambda r:float(np.linalg.norm(r)/np.sqrt(len(r)+1))
        badHistory=[True]*4;terrible=0;reason='initial';stop='max_iterations';converged=False
        completionVerification=Struct(Applied=False)
        for iteration in range(maximum+1):
            self.checkCancel();error=rms(residual);inf=float(np.max(abs(residual),initial=0))
            record=Struct(Iteration=iteration,ResidualInf=inf,ResidualRms=error,
                ElapsedSeconds=time.perf_counter()-started,JacobianBuilds=self.counts.JacobianBuilds,
                FactorReuses=self.counts.FactorReuses)
            if o.get('ProgressFcn'):o['ProgressFcn'](record)
            if o.get('Display','iter') not in ('none','off'):
                print(f'稀疏求解迭代 {iteration}：残差无穷范数={inf:.8e}，均方根={error:.8e}',flush=True)
            if inf<=tol:
                # 独立终态复核不改变用户容差，也不挤占已经用尽的迭代预算。
                if inf>0 and iteration<maximum:
                    state,residual,completionVerification=self._verifyConverged(state,residual,intervals,tol)
                converged=True;stop='common_max_norm_tolerance';break
            if iteration==maximum:break
            rebuilt=self.factor.factor is None;js=0.;beforeLinear=self.factor.factorSeconds+self.factor.solveSeconds
            if rebuilt:js=self._build(state,intervals)
            else:self.counts.FactorReuses+=1
            precisionStepsBefore=len(self.factor.regularizedSteps)
            target=targetPolicy.calculate(residual);step=self.factor.solve(-target)
            t=time.perf_counter()
            candidate,new,alpha,trials=self._trial(state,step)
            trialSeconds=time.perf_counter()-t
            if rebuilt and alpha<targetPolicy.fraction and self.factor.precisionLimited():
                # 极弱方向已低于 float64 分辨率：原 RHS 不变，仅恢复可辨识的线性步。
                recoveryStarted=time.perf_counter()
                step=self.factor.regularizedSolve(-target)
                self.counts.PrecisionRecoverySeconds+=time.perf_counter()-recoveryStarted
                t=time.perf_counter();candidate,new,alpha,extra=self._trial(state,step);trials+=extra
                trialSeconds+=time.perf_counter()-t
                # 该步只保证线性子问题的最小改变量，再以实际原方程残差决定缩步。
                while new is not None and rms(new)>=error:
                    t=time.perf_counter();candidate,new,alpha,extra=self._trial(state,step,alpha*.5);trials+=extra
                    trialSeconds+=time.perf_counter()-t
            if new is None:stop='no_feasible_trial';break
            accepted=rms(new)<error or np.max(abs(new),initial=0)<=tol
            if not accepted and not rebuilt:
                # 旧分解无法改善时撤销，下一轮重建。
                candidate=state;new=residual;alpha=0.
            elif not accepted:
                # 新分解可在再减半后接受非单调步。
                t=time.perf_counter();candidate,new,alpha,extra=self._trial(state,step,alpha*.5);trials+=extra
                trialSeconds+=time.perf_counter()-t
                if new is None:stop='no_acceptable_fresh_trial';break
                accepted=True
            self.counts.LineSearchSeconds+=trialSeconds
            # 按原始目标变化检查新旧线性化，不把实际残差按缩步量改小。
            predictedChange=target
            prediction=residual-predictedChange
            defect=float(np.linalg.norm(new-prediction)/max(np.linalg.norm(predictedChange),np.finfo(float).tiny))
            threshold=.7 if error>.1 else .7/max(abs(.434*np.log(max(error,np.finfo(float).tiny))),np.finfo(float).tiny)
            bad=defect>=threshold or not accepted
            badHistory[(iteration+1)%4]=bad
            row=Struct(Iteration=iteration+1,ResidualInf=float(np.max(abs(new),initial=0)),ResidualRms=rms(new),
                Accepted=accepted,LineSearchAlpha=alpha,LineTrials=trials,JacobianRebuilt=rebuilt,
                RebuildReason=reason if rebuilt else '',LinearizationDefect=defect,LinearizationThreshold=threshold,
                LimitedRows=int(np.count_nonzero(target!=residual)),JacobianSeconds=js,
                LinearSolveSeconds=self.factor.factorSeconds+self.factor.solveSeconds-beforeLinear,
                RawStepInf=float(np.max(abs(step),initial=0)),ElapsedSeconds=time.perf_counter()-started,
                PrecisionRecovery=bool(len(self.factor.regularizedSteps)>precisionStepsBefore))
            history.append(row)
            if accepted:state=candidate;residual=new
            if bad:
                self.factor.invalidate();reason='linear_prediction_inaccurate' if accepted else 'reused_factor_rejected'
                if (iteration+1)%4==1 and all(badHistory[:3]):
                    terrible+=1
                    # 用对称曲率探测更新每列差分间隔。
                    intervalStarted=time.perf_counter()
                    intervals=updateIntervals(self.worker,state,intervals,self.vs,self.rs)
                    self.counts.FDUpdateSeconds+=time.perf_counter()-intervalStarted
                    self.counts.FunctionEvaluations+=2*len(state)
                    self.counts.FDIntervalUpdates+=1;row.FDIntervalUpdated=True
                if terrible>5:stop='sustained_bad_linearization';break
        # 独立、无缓存原方程验收；不使用目标残差和派生缓存作为停止证据。
        checked=self.reference.evaluateResidual(state*self.vs)/self.rs
        final=float(np.max(abs(checked),initial=0))
        converged=bool(converged and final<=tol)
        info=Struct(Converged=converged,converged=converged,Iterations=len(history),
            FinalResidualInf=final,FinalResidualRms=rms(checked),History=history,StopReason=stop,
            ElapsedSeconds=time.perf_counter()-started,UnknownCount=len(state),
            Solver='Sparse_AFC_AVC_float64',Profile='adaptive_factor_cache_precision_recovery',
            LinearAlgebraBackend='SuperLU_COLAMD_float64',Tolerance=tol,StepLevel=level,
            KeyFraction=targetPolicy.fraction,KeyRows=len(self.policies.keyRows),
            PositiveRatioLimit=o.get('PositiveRatioLimit',True),
            FiniteDifferenceMin=float(intervals.min()),FiniteDifferenceMax=float(intervals.max()),
            LinearSolveSeconds=self.factor.factorSeconds+self.factor.solveSeconds,
            NumericalFactorizations=self.factor.factorizations,TriangularSolves=self.factor.solves,
            PrecisionLimitedSteps=self.factor.regularizedSteps,
            CompletionVerification=completionVerification,
            PatternNonzeros=sum(len(rows) for rows in self.worker.runtime.plan.affectedRows),NoColoring=True,NoGMRES=True,
            IndependentResidualVerification=True,RuntimeStats=dict(self.worker.runtime.stats),
            ModelFingerprint=self.modelFingerprint,
            DependencyValues=self.worker.runtime.scalarCount,DependencyGranularity=self.worker.runtime.granularity,
            NativeGridPrograms=[self.worker.runtime.describe()],
            FidelityLimits=['数值运算采用 float64 和 SuperLU',
                '关键残差限幅采用求解期间的运行峰值',
                '病态线性步采用条件估计与 LSMR 恢复',
                '达标后的完整 Newton 精度复核属于当前 float64 后端适配',
                '不透明扩展接口保留声明的保守依赖'])
        info.update(self.counts)
        return state,info


# 维护数值状态、依赖或计算接口。
def solve(problem,initial=None,options=None):
    """以唯一的当前算法求解方形实数方程组，允许直接使用问题登记初值。"""
    return NonlinearSolver(problem,{} if options is None else options).solve(problem.initialState if initial is None else initial)
