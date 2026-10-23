"""逐列 AFC 有限差分；列之间相互独立，单进程节点调度，无着色。"""
import os
import numpy as np
from scipy import sparse


# 维护数值状态、依赖或计算接口。
class FiniteDifferenceWorker:
    # 初始化数值对象。
    def __init__(self,problem):
        self.problem=problem
        self.runtime=problem.createRuntime()


    # 维护数值状态、依赖或计算接口。
    def intervals(self,task,variableScale,residualScale):
        """对称探测曲率，根据舍入误差与曲率平衡更新每列差分间隔。"""
        roundoff=2000*np.finfo(float).eps/2
        probe=2*(3*roundoff)**.25
        positive=dict(task,steps=np.full(len(task['columns']),probe),dense=True)
        negative=dict(task,steps=np.full(len(task['columns']),-probe),dense=True)
        plus=self.columns(positive,variableScale,residualScale)
        minus=self.columns(negative,variableScale,residualScale)
        baseline=self.runtime.residual()/residualScale
        threshold=100*roundoff/probe**2
        values=[];offset=0
        for column,length,hp,hm,old in zip(task['columns'],plus['lengths'],plus['steps'],minus['steps'],task['steps']):
            take=slice(offset,offset+length);rows=plus['rows'][take];offset+=length
            interval=old
            if hp*hm<0 and abs(abs(hp)-abs(hm))<1e-12*max(abs(hp),abs(hm)):
                curvature=abs((plus['data'][take]-minus['data'][take])/hp)
                selected=curvature>threshold
                if np.any(selected):
                    optimum=2*np.sqrt(roundoff*np.maximum(1,abs(baseline[rows][selected]))/curvature[selected])
                    interval=float(np.exp(np.mean(np.log(optimum))))
                else:interval=2*np.sqrt(roundoff/threshold)
            values.append(interval)
        return dict(columns=task['columns'],intervals=values)

    # 维护数值状态、依赖或计算接口。
    def columns(self,task,variableScale,residualScale):
        runtime=self.runtime;base=np.asarray(task['state'])*variableScale
        baseline=runtime.accept(base)
        w=np.asarray(task['state']);lower=self.problem.lowerBound/variableScale;upper=self.problem.upperBound/variableScale
        data=[];rowParts=[];lengths=[];steps=[];reversals=reductions=0
        for col,nominal in zip(task['columns'],task['steps']):
            rows=np.asarray(runtime.plan.affectedRows[col],int)
            success=False
            for reduction in range(18):
                for sign in (1.,-1.):
                    h=sign*nominal*2.**(-reduction)
                    candidate=w[col]+h
                    if not lower[col]<=candidate<=upper[col] or candidate==w[col]:continue
                    state=base.copy();state[col]=candidate*variableScale[col]
                    try:
                        with runtime.transaction(state):value=runtime.residual(rows,runtime.columnProviders[col])
                    except Exception as error:
                        if self.problem.isDomainError(error):continue
                        raise
                    derivative=(value-baseline[rows])/residualScale[rows]/h
                    if not np.isfinite(derivative).all():raise ValueError('nonlinear:NonfiniteJacobian')
                    keep=np.arange(len(rows)) if task.get('dense',False) else np.flatnonzero(derivative!=0)
                    data.extend(derivative[keep]);rowParts.extend(rows[keep]);lengths.append(len(keep));steps.append(h)
                    reversals+=int(sign<0);reductions+=reduction;success=True;break
                if success:break
            if not success:raise ValueError(f'nonlinear:FiniteDifferenceDomain: column {col}')
        return dict(data=np.asarray(data),rows=np.asarray(rowParts,int),lengths=lengths,
            columns=task['columns'],steps=steps,reversals=reversals,reductions=reductions,
            runtimeStats=dict(runtime.stats),propertyStats=dict(runtime.propertyStats),pid=os.getpid())


# 维护数值状态、依赖或计算接口。
def buildJacobian(worker,state,stepSizes,variableScale,residualScale,chunk=32,cancel=None):
    """每列从同一接受态开始；所有列用静态潜在依赖，不删数值小项。"""
    tasks=[dict(kind='jacobianColumns',state=state,columns=list(range(lo,min(lo+chunk,len(state)))),
        steps=stepSizes[lo:lo+chunk]) for lo in range(0,len(state),chunk)]
    parts=[]
    for task in tasks:
        if cancel and cancel():raise InterruptedError('nonlinear:Cancelled')
        parts.append(worker.columns(task,variableScale,residualScale))
    data=np.concatenate([part['data'] for part in parts]);rows=np.concatenate([part['rows'] for part in parts])
    cols=np.concatenate([np.repeat(part['columns'],part['lengths']) for part in parts])
    matrix=sparse.csc_matrix((data,(rows,cols)),shape=(len(state),len(state)))
    matrix.eliminate_zeros();matrix.sort_indices()
    return matrix,parts


# 维护数值状态、依赖或计算接口。
def updateIntervals(worker,state,intervals,variableScale,residualScale):
    tasks=[dict(kind='jacobianIntervals',state=state,columns=list(range(lo,min(lo+32,len(state)))),
        steps=intervals[lo:lo+32]) for lo in range(0,len(state),32)]
    parts=[worker.intervals(t,variableScale,residualScale) for t in tasks]
    result=intervals.copy()
    for part in parts:result[part['columns']]=part['intervals']
    return result
