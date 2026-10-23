"""通用标量求值上下文：每次求解独占状态、缓存、参考态和 journal 事务。"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from nonlinearSolver.dependencyGraph import EvaluationPlan

# 维护数值状态、依赖或计算接口。
@dataclass(frozen=True)
class ScalarSpan:
    first:int
    count:int
    name:str
    shape:tuple
    components:int


# 维护数值状态、依赖或计算接口。
class ScalarRuntime:
    granularity='global scalar dependencies; direct AVC adjacency'
    # 初始化数值对象。
    def __init__(self,initial,program):
        self.state=np.asarray(initial,float).copy();self.stats=Counter();self.inTransaction=False
        self.rollbackPolicy='journal';self.program=program;self.p=program
        self.root=program.root('unknowns',self.state)
        self.outputs={};self.outputRows={};self.conservativePorts=[]

    # 维护数值状态、依赖或计算接口。
    @property
    def propertyStats(self):
        """回调适配器可提供统计；普通函数可返回空统计。"""
        return {}

    # 维护数值状态、依赖或计算接口。
    def _freeze(self):
        n=len(self.state);affected=[set() for _ in range(n)];providers=[[] for _ in range(n)]
        for name,f in self.outputs.items():
            for row,bits in zip(self.outputRows[name],f.deps):
                providers[row].append(name)
                while bits:
                    bit=bits&-bits;affected[bit.bit_length()-1].add(int(row));bits-=bit
        missing=[i for i,ports in enumerate(providers) if not ports]
        if missing:raise ValueError('nonlinear:UnprovidedScalarEquations: '+str(missing[:20]))
        self.spans=[];offset=0
        for node in self.p.nodes:
            self.spans.append(ScalarSpan(offset,node.value.size,node.label,(len(node.deps),),node.value.shape[0]));offset+=node.value.size
        self.specs=tuple(self.spans);self.scalarCount=offset
        self.plan=EvaluationPlan(self.specs,(),tuple(tuple(sorted(v)) for v in affected),tuple(tuple(v) for v in providers),n)
        self.columnProviders=tuple(None for _ in range(n))
        # 未连接的给定边界可形成零未知量网络，空方程集保留明确的空索引形状。
        self.outputOrder=np.concatenate(list(self.outputRows.values())) if self.outputRows else np.empty(0,dtype=int)
        self.uniqueRows=len(np.unique(self.outputOrder))==len(self.outputOrder)
        targets=np.asarray([(f.node,f.component,i) for f in self.outputs.values() for i in range(f.value.size)],dtype=np.int32).reshape(-1,3)
        order=np.argsort(self.outputOrder,kind='stable');self.rowTargets=targets[order]
        self.rowOffsets=np.r_[0,np.cumsum(np.bincount(self.outputOrder,minlength=n))]

    # 维护数值状态、依赖或计算接口。
    def residual(self,rows=None,providers=None):
        selected=np.arange(len(self.state)) if rows is None else np.asarray(rows,int)
        if self.uniqueRows:targets=self.rowTargets[selected]
        else:
            ids=np.concatenate([np.arange(self.rowOffsets[i],self.rowOffsets[i+1]) for i in selected]) if len(selected) else np.empty(0,int)
            targets=self.rowTargets[ids]
        r=self.p.run(dict(unknowns=self.state),targets=targets)
        if not self.uniqueRows:
            counts=np.diff(self.rowOffsets)[selected];r=np.add.reduceat(r,np.r_[0,np.cumsum(counts)[:-1]]) if len(selected) else r
        self.stats['residualRequests']+=1
        if not np.all(np.isfinite(r)):raise ValueError('nonlinear:NonfiniteScalarResidual')
        return r

    # 维护数值状态、依赖或计算接口。
    def accept(self,state):
        self.state=np.asarray(state,float).copy();result=self.residual()
        return result

    # 维护数值状态、依赖或计算接口。
    @contextmanager
    def transaction(self,state):
        if self.inTransaction:raise RuntimeError('nonlinear:NestedScalarTransaction')
        if self.rollbackPolicy=='journal':
            if not self.p.started:self.residual()
            self.p.begin()
        saved=self.state;self.state=np.asarray(state,float).copy();self.inTransaction=True
        try:yield self
        finally:
            if self.rollbackPolicy=='journal':self.p.rollback()
            self.state=saved;self.inTransaction=False;self.stats['rollbacks']+=1

    # 维护数值状态、依赖或计算接口。
    def describe(self):
        out=self.p.describe();out.update(global_scalar_ids=self.scalarCount,conservative_ports=self.conservativePorts,rollback=self.rollbackPolicy)
        return out
