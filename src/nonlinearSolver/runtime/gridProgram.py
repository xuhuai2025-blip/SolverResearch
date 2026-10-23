"""空间/相位节点图：终端依赖在建图时声明，由本地 C++ 调度失效点。

依赖在构建数值图时显式声明，运行时不解析源码。
支持数组、周期微分、平均、谐波与自定义 Python 回调。
"""
from collections import Counter, OrderedDict
import numpy as np
from nonlinearSolver.values import Struct


# 维护数值状态、依赖或计算接口。
class Field:
    __array_priority__=1000
    # 初始化数值对象。
    def __init__(self,p,node,component,shape):self.p=p;self.node=node;self.component=component;self.shape=shape
    # 维护数值状态、依赖或计算接口。
    @property
    def value(self):return self.p.nodes[self.node].value[self.component].reshape(self.shape)
    # 维护数值状态、依赖或计算接口。
    @property
    def deps(self):return self.p.nodes[self.node].deps
    # 维护数值状态、依赖或计算接口。
    def __getitem__(self,key):
        ids=np.arange(np.prod(self.shape)).reshape(self.shape)[key]
        return self.p.gather(self,np.asarray(ids).reshape(np.shape(ids) or (1,1)))
    # 维护数值状态、依赖或计算接口。
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
        return self.p.gather(self,np.arange(np.prod(self.shape)).reshape(*shape))
    # 维护数值状态、依赖或计算接口。
    def __add__(self,b):return self.p.point(np.add,self,b)
    # 维护数值状态、依赖或计算接口。
    def __radd__(self,b):return self.p.point(np.add,b,self)
    # 维护数值状态、依赖或计算接口。
    def __sub__(self,b):return self.p.point(np.subtract,self,b)
    # 维护数值状态、依赖或计算接口。
    def __rsub__(self,b):return self.p.point(np.subtract,b,self)
    # 维护数值状态、依赖或计算接口。
    def __mul__(self,b):return self.p.point(np.multiply,self,b)
    # 维护数值状态、依赖或计算接口。
    def __rmul__(self,b):return self.p.point(np.multiply,b,self)
    # 维护数值状态、依赖或计算接口。
    def __truediv__(self,b):return self.p.point(np.divide,self,b)
    # 维护数值状态、依赖或计算接口。
    def __rtruediv__(self,b):return self.p.point(np.divide,b,self)
    # 维护数值状态、依赖或计算接口。
    def __pow__(self,b):return self.p.point(np.power,self,b)
    # 维护数值状态、依赖或计算接口。
    def __neg__(self):return self.p.point(np.negative,self)
    # 维护数值状态、依赖或计算接口。
    def __abs__(self):return self.p.point(np.absolute,self)
    # 维护数值状态、依赖或计算接口。
    def __lt__(self,b):return self.p.point(np.less,self,b)
    # 维护数值状态、依赖或计算接口。
    def __le__(self,b):return self.p.point(np.less_equal,self,b)
    # 维护数值状态、依赖或计算接口。
    def __gt__(self,b):return self.p.point(np.greater,self,b)
    # 维护数值状态、依赖或计算接口。
    def __ge__(self,b):return self.p.point(np.greater_equal,self,b)
    # 维护数值状态、依赖或计算接口。
    @property
    def real(self):return self.p.point(np.real,self)
    # 维护数值状态、依赖或计算接口。
    @property
    def imag(self):return self.p.point(np.imag,self)

# 维护数值状态、依赖或计算接口。
class GridProgram:
    """静态逐点依赖图，只批量重算失效的标量条目。"""
    # 初始化数值对象。
    def __init__(self,label,property_cache=None):
        self.label=label;self.nodes=[];self.roots={};self.root_count=0;self.constants={};self.outputs={};self.stats=Counter();self.started=False
        self.property_cache=property_cache if property_cache is not None else OrderedDict()
        self.checkpoint=None;self.sealed=False
    # 维护数值状态、依赖或计算接口。
    def root(self,name,value):
        if self.sealed:raise RuntimeError('nonlinear:grid: 依赖图已冻结，不能修改')
        a=np.asarray(value);shape=a.shape or (1,1);n=a.size
        deps=[1<<(self.root_count+i) for i in range(n)]
        node=Struct(label=name,value=np.asarray(a).reshape(1,-1).copy(),deps=deps,parents=[],maps=[],fn=None,root=True)
        self.nodes.append(node);f=Field(self,len(self.nodes)-1,0,shape)
        self.roots[name]=(f,np.arange(self.root_count,self.root_count+n));self.root_count+=n
        return f
    # 维护数值状态、依赖或计算接口。
    def constant(self,value):
        a=np.asarray(value);key=(a.shape,a.dtype.str,a.tobytes())
        if key not in self.constants:
            if self.sealed:raise RuntimeError('nonlinear:grid: 依赖图已冻结，不能修改')
            self.nodes.append(Struct(label='constant',value=a.reshape(1,-1).copy(),deps=[0]*a.size,parents=[],maps=[],fn=None,root=False))
            self.constants[key]=Field(self,len(self.nodes)-1,0,a.shape or (1,1))
        return self.constants[key]
    # 维护数值状态、依赖或计算接口。
    def field(self,a):return a if isinstance(a,Field) else self.constant(a)
    # 维护数值状态、依赖或计算接口。
    def op(self,label,parents,maps,shape,fn,components=1,dtype=float,*,nativeOperation=None):
        """登记回调和依赖；展示标签不决定计算方式，内置算子另行显式登记类型。"""
        if self.sealed:raise RuntimeError('nonlinear:grid: 依赖图已冻结，不能修改')
        parents=[self.field(p) for p in parents];n=int(np.prod(shape));deps=[]
        for i in range(n):
            d=0
            for p,m in zip(parents,maps):
                for j in np.asarray(m[i]).ravel():d |= p.deps[int(j)]
            deps.append(d)
        self.nodes.append(Struct(label=label,nativeOperation=nativeOperation,value=np.zeros((components,n),dtype=dtype),deps=deps,parents=parents,maps=maps,fn=fn,root=False))
        fields=[Field(self,len(self.nodes)-1,i,shape) for i in range(components)]
        return fields[0] if components==1 else fields
    # 维护数值状态、依赖或计算接口。
    def point(self,fn,*args,dtype=None):
        fs=[self.field(a) for a in args];shape=np.broadcast_shapes(*(f.shape for f in fs))
        maps=[np.broadcast_to(np.arange(np.prod(f.shape)).reshape(f.shape),shape).ravel()[:,None] for f in fs]
        if dtype is None:
            with np.errstate(all='ignore'):
                dtype=np.asarray(fn(*[np.ones(1,dtype=f.value.dtype) for f in fs])).dtype
        # 只按受控 NumPy 函数的对象身份启用原生算术；同名自定义回调仍执行自身函数体。
        nativeFunctions=((np.add,'add'),(np.subtract,'subtract'),(np.multiply,'multiply'),
            (np.divide,'divide'),(np.power,'power'),(np.negative,'negative'),(np.absolute,'absolute'),
            (np.real,'real'),(np.imag,'imag'),(np.less,'less'),(np.less_equal,'less_equal'),
            (np.greater,'greater'),(np.greater_equal,'greater_equal'),(np.equal,'equal'),
            (np.where,'where'),(np.maximum,'maximum'),(np.minimum,'minimum'),(np.sqrt,'sqrt'),
            (np.log,'log'),(np.tanh,'tanh'))
        operation=next((name for function,name in nativeFunctions if fn is function),None)
        return self.op(getattr(fn,'__name__','point'),fs,maps,shape,lambda ids,*xs:fn(*[x[:,0] for x in xs]),
                       dtype=dtype,nativeOperation=operation)
    # 维护数值状态、依赖或计算接口。
    def gather(self,f,ids):
        # 索引映射由该内置构造器拥有，独立于可修改的展示标签。
        return self.op('gather',[f],[ids.ravel()[:,None]],ids.shape,lambda changed,x:x[:,0],
                       dtype=f.value.dtype,nativeOperation='gather')
    # 维护数值状态、依赖或计算接口。
    def where(self,condition,a,b):return self.point(np.where,condition,a,b)
    # 维护数值状态、依赖或计算接口。
    def maximum(self,a,b):return self.point(np.maximum,a,b)
    # 维护数值状态、依赖或计算接口。
    def minimum(self,a,b):return self.point(np.minimum,a,b)
    # 维护数值状态、依赖或计算接口。
    def sqrt(self,a):return self.point(np.sqrt,a)
    # 维护数值状态、依赖或计算接口。
    def cat(self,*fs):
        fs=[self.field(f) for f in fs];fs=[f for f in fs if np.prod(f.shape)];nt=fs[0].shape[1];shape=(sum(f.shape[0] for f in fs),nt)
        # 显式拼接的每个输出点只读取所属的一个父输入。
        selectors=[];maps=[];offset=0
        for f in fs:
            mask=np.zeros(np.prod(shape),bool);mask[offset:offset+np.prod(f.shape)]=True;selectors.append(mask)
            idx=np.zeros(np.prod(shape),int);idx[mask]=np.arange(np.prod(f.shape));maps.append(idx[:,None]);offset+=np.prod(f.shape)
        # 显式登记原生拼接，再替换依赖集合，排除不属于该输出段的父输入。
        out=self.op('concatenate',fs,maps,shape,lambda ids,*xs:sum(np.where(mask[ids],x[:,0],0) for mask,x in zip(selectors,xs)),
                    dtype=np.result_type(*(f.value.dtype for f in fs)),nativeOperation='concatenate')
        node=self.nodes[out.node];node.deps=[0]*np.prod(shape);node.selector=np.argmax(np.asarray(selectors),axis=0)
        offset=0
        for f in fs:node.deps[offset:offset+np.prod(f.shape)]=f.deps;offset+=np.prod(f.shape)
        return out
    # 维护数值状态、依赖或计算接口。
    def spatial(self,matrix,f):
        matrix=matrix.tocsr();nt=f.shape[1];width=max(np.diff(matrix.indptr),default=0);width=max(1,width)
        ids=np.zeros((matrix.shape[0],width),int);weights=np.zeros_like(ids,dtype=float)
        for i in range(matrix.shape[0]):
            a,b=matrix.indptr[i:i+2];ids[i,:b-a]=matrix.indices[a:b];weights[i,:b-a]=matrix.data[a:b]
        shape=(matrix.shape[0],nt);rr,tt=np.indices(shape);maps=ids[rr.ravel()]*nt+tt.ravel()[:,None];ww=weights[rr.ravel()]
        # 维护数值状态、依赖或计算接口。
        def calc(changed,x):
            # 按 CSR 列表的升序顺序累加，保持浮点求和顺序稳定。
            y=np.zeros(len(changed),dtype=x.dtype)
            for j in range(width):y+=ww[changed,j]*x[:,j]
            return y
        # 空间模板由映射和权重共同定义，不从展示名推断计算类型。
        out=self.op('space stencil',[f],[maps],shape,calc,dtype=f.value.dtype,nativeOperation='space stencil')
        self.nodes[out.node].weights=ww
        # 零填充不构成实际依赖。
        self.nodes[out.node].deps=[self._union(f.deps[int(j)] for j,w in zip(row,weight) if w!=0) for row,weight in zip(maps,ww)]
        return out
    # 维护数值状态、依赖或计算接口。
    @staticmethod
    def _union(values):
        v=0
        for x in values:v|=x
        return v
    # 维护数值状态、依赖或计算接口。
    def mean(self,f):
        n,nt=f.shape;maps=np.arange(n*nt).reshape(n,nt)
        return self.op('period mean',[f],[maps],(n,1),lambda ids,x:np.mean(x,axis=1),dtype=f.value.dtype)
    # 维护数值状态、依赖或计算接口。
    def announce(self,outputs):
        if self.sealed:raise RuntimeError('nonlinear:grid: 依赖图已冻结，不能修改')
        self.outputs=outputs
        # AFC 的汇点由所需方程决定。只删无汇点可达路径的节点，
        # where 的两个潜在分支以及跨相位父节点都必须保留。
        self.live=set();pending=[field.node for field in outputs.values()]
        while pending:
            node=pending.pop()
            if node in self.live:continue
            self.live.add(node);pending.extend(f.node for f in self.nodes[node].parents)
        self.computations=[i for i,node in enumerate(self.nodes) if node.fn is not None and i in self.live]
        self.afc={name:[int(bits.bit_count()) for bits in field.deps] for name,field in outputs.items()}
        self.outputTargets=np.asarray([(f.node,f.component,i) for f in self.outputs.values() for i in range(f.value.size)],dtype=np.int32).reshape(-1,3)
        from nonlinearSolver.runtime.nativeGrid import NativeGrid
        self.native=NativeGrid(self)
        self.last_outputs=None
        self.outputViews={name:field.value for name,field in self.outputs.items()}
        self.sealed=True
    # 维护数值状态、依赖或计算接口。
    def begin(self):
        if self.checkpoint is not None:raise RuntimeError('nonlinear:grid: 不允许嵌套事务')
        self.checkpoint=([(f,f.value.copy()) for f,ids in self.roots.values()],self.started,self.last_outputs)
        self.native.begin()
    # 维护数值状态、依赖或计算接口。
    def rollback(self):
        roots,started,outputs=self.checkpoint
        self.native.rollback()
        for f,value in roots:self.nodes[f.node].value[f.component]=value.ravel()
        self.started=started;self.last_outputs=outputs;self.checkpoint=None
    # 维护数值状态、依赖或计算接口。
    def run(self,values,copyOutputs=True,targets=None):
        changed=[]
        for name,(f,ids) in self.roots.items():
            a=np.asarray(values[name]);flat=np.asarray(a,dtype=f.value.dtype).ravel()
            if a.shape!=(f.shape if a.shape else ()) and a.size!=np.prod(f.shape):raise ValueError('nonlinear:grid: 数组形状发生变化 '+name)
            bits=flat.view(np.uint64);old=f.value.ravel().view(np.uint64)
            if np.array_equal(bits,old):continue
            different=bits!=old
            if np.iscomplexobj(flat):different=np.any(different.reshape(-1,2),axis=1)
            local=np.flatnonzero(different);changed.extend(ids[local].tolist())
            self.nodes[f.node].value[0]=flat
        if self.started and not changed and targets is None:
            self.stats['unchanged_root_reuses']+=1
            return {name:value.copy() for name,value in self.outputViews.items()} if copyOutputs else self.outputViews
        if self.native is not None:
            reset=not self.started;self.started=False
            selected=self.native.run(changed,reset,self.outputTargets if targets is None else targets,read=targets is not None)
            self.started=True
            self.stats['runs']+=1;self.stats['changed_terminal_entries']+=len(changed)
            if targets is not None:return selected
            if not copyOutputs:
                self.last_outputs=None;return self.outputViews
            self.last_outputs={name:field.value.copy() for name,field in self.outputs.items()}
            return dict(self.last_outputs)
    # 维护数值状态、依赖或计算接口。
    def describe(self):
        return dict(label=self.label,scheduler='compiled',linearization='direct',terminal_entries=self.root_count,smart_value_entries=sum(n.value.size for n in self.nodes if n.fn is not None),calculators=len(self.computations),afc_sink_entries=int(sum(np.prod(f.shape) for f in self.outputs.values())),compiled_scheduler=self.native is not None,stats=dict(self.stats),native_stats=self.native.stats() if self.native else {})
