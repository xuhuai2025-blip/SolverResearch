"""原生逐标量调度绑定：校验跨平台 C ABI 后绑定数值图接口。"""
import ctypes as ct
import hashlib
from nonlinearSolver.runtime.nativeLibrary import selectedPath, validateLibraryPath, hostTarget
import numpy as np

I=ct.c_int;P=ct.c_void_p
# 维护数值状态、依赖或计算接口。
class Parent(ct.Structure):_fields_=[('node',I),('component',I),('width',I),('map',P)]
# 维护数值状态、依赖或计算接口。
class Node(ct.Structure):_fields_=[('opcode',I),('size',I),('components',I),('iscomplex',I),('parent_count',I),('value',P),('parents',P),('weights',P),('selector',P)]
# 维护数值状态、依赖或计算接口。
class Edge(ct.Structure):_fields_=[('node',I),('point',I)]
CALLBACK=ct.CFUNCTYPE(I,I,I,ct.POINTER(I))
# 维护数值状态、依赖或计算接口。
class Target(ct.Structure):_fields_=[('node',I),('component',I),('point',I)]
_libraries={}


# 维护数值状态、依赖或计算接口。
def library():
    """按所选绝对路径缓存库句柄；已创建的数值图不会切换到另一份动态库。"""
    path=selectedPath();key=str(path)
    if key in _libraries:return _libraries[key]
    path=validateLibraryPath(path)
    try:
        lib=ct.CDLL(str(path))
        lib.grid_abi_version.argtypes=[];lib.grid_abi_version.restype=I
        lib.grid_abi_layout.argtypes=[I];lib.grid_abi_layout.restype=I
        if lib.grid_abi_version()!=2:raise ValueError('调度库接口版本不匹配，请重新构建本版库')
        # 指针、复数和结构体字段都先核对，禁止把不兼容布局交给原生函数。
        expected=[ct.sizeof(Parent),ct.sizeof(Node),ct.sizeof(Edge),ct.sizeof(Target),ct.sizeof(P),ct.sizeof(I),ct.sizeof(ct.c_int64),ct.sizeof(ct.c_double),16,
                  Parent.map.offset,Node.value.offset,Node.parents.offset,Node.weights.offset,Node.selector.offset]
        if [lib.grid_abi_layout(i) for i in range(len(expected))]!=expected:
            raise ValueError('调度库内存布局与当前 Python 不兼容')
        definitions={'grid_create_direct':([P,I,P,P],P),
            'grid_run':([P,I,P,CALLBACK],I),'grid_request':([P,I,P,I,P,P,CALLBACK],I),
            'grid_clear':([P],None),'grid_delete':([P],None),'grid_stats':([P,P],None),
            'grid_begin':([P],None),'grid_rollback':([P],None),'grid_stats_detail':([P,P],None),
            'grid_compile_targets':([P,I,P],None)}
        for name,(args,result) in definitions.items():
            fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    except (OSError,AttributeError) as exc:
        raise ValueError('无法加载本版原生调度接口，请选择匹配的调度库：'+str(exc)) from exc
    target=hostTarget()
    lib.scheduler_info=dict(platform=target['platform'],architecture=target['architecture'],abiVersion=2,
                            filename=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    _libraries[key]=lib
    return lib


# 维护数值状态、依赖或计算接口。
def libraryInfo():
    """结果保留实际库的名称、架构和字节身份，不把本机路径写入问题定义。"""
    return dict(library().scheduler_info)

# 维护数值状态、依赖或计算接口。
class NativeGrid:
    # 初始化数值对象。
    def __init__(self,program):
        self.program=program;self.lib=library();self.keep=[];self.nodes=(Node*len(program.nodes))();self.error=None
        codes={'add':1,'subtract':2,'multiply':3,'divide':4,'power':5,'negative':6,'absolute':7,'real':8,'imag':9,'less':10,'less_equal':11,'greater':12,'greater_equal':13,'equal':14,'where':15,'maximum':16,'minimum':17,'sqrt':18,'log':19,'tanh':20,'gather':21,'space stencil':22,'concatenate':23}
        for i,node in enumerate(program.nodes):
            node.value=np.ascontiguousarray(node.value,dtype=complex if np.iscomplexobj(node.value) else float)
            parents=(Parent*len(node.parents))();self.keep.append(parents)
            for j,(f,m) in enumerate(zip(node.parents,node.maps)):
                m=np.ascontiguousarray(m,dtype=np.int64);node.maps[j]=m;self.keep.append(m)
                parents[j]=Parent(f.node,f.component,m.shape[1],m.ctypes.data)
            weights=np.ascontiguousarray(node.get('weights',[]),dtype=float);selector=np.ascontiguousarray(node.get('selector',[]),dtype=np.int64);self.keep.extend([weights,selector])
            # 只消费构造器显式登记的运算类型；任意展示名和自定义函数名均不能替换回调语义。
            opcode=-1 if node.fn is None or i not in program.live else codes.get(node.nativeOperation,0)
            # 复数幂、除法和超越函数保留 NumPy 算法，避免不同 C++ 数学库
            # 在大热穿透参数处产生有限差分不一致。
            if opcode in (4,5,18,19,20) and (np.iscomplexobj(node.value) or any(np.iscomplexobj(f.value) for f in node.parents)):opcode=0
            self.nodes[i]=Node(opcode,len(node.deps),node.value.shape[0],int(np.iscomplexobj(node.value)),len(node.parents),node.value.ctypes.data,ct.addressof(parents),weights.ctypes.data,selector.ctypes.data)
        entries=[];offsets=[0]
        # 根索引只映射到根节点标量；下游邻接由 C++ 直接父边建立。
        for field,ids in program.roots.values():
            for j in range(len(ids)):entries.append((field.node,j));offsets.append(len(entries))
        self.edges=np.asarray(entries,dtype=np.int32).reshape(-1,2);self.offsets=np.asarray(offsets,dtype=np.int64)
        self.handle=self.lib.grid_create_direct(ct.addressof(self.nodes),len(self.nodes),self.offsets.ctypes.data,self.edges.ctypes.data)
        if not self.handle:raise RuntimeError('原生调度器创建失败')
        targets=program.outputTargets
        self.lib.grid_compile_targets(self.handle,len(targets),targets.ctypes.data)
        self.callback=CALLBACK(self.calculate)
    # 维护数值状态、依赖或计算接口。
    def calculate(self,i,count,ids):
        try:
            ids=np.ctypeslib.as_array(ids,shape=(count,));node=self.program.nodes[i]
            args=[f.value.ravel()[m[ids]] for f,m in zip(node.parents,node.maps)]
            with np.errstate(all='ignore'):value=np.asarray(node.fn(ids,*args)).reshape(node.value.shape[0],count)
            node.value[:,ids]=value
            self.program.stats['opaque_numpy_batches']+=1;self.program.stats['opaque_values_calculated']+=value.size
            return 1
        except BaseException as error:self.error=error;return 0
    # 维护数值状态、依赖或计算接口。
    def run(self,changed,reset,targets=None,read=False):
        if reset:self.lib.grid_clear(self.handle)
        changed=np.asarray(changed,dtype=np.int64)
        output=None
        if targets is None:
            result=self.lib.grid_run(self.handle,len(changed),changed.ctypes.data,self.callback)
        else:
            targets=np.ascontiguousarray(targets,dtype=np.int32).reshape(-1,3)
            if read:output=np.empty(len(targets),dtype=float)
            result=self.lib.grid_request(self.handle,len(changed),changed.ctypes.data,len(targets),targets.ctypes.data,output.ctypes.data if read else None,self.callback)
        if result!=1:
            error=self.error;self.error=None
            if error is not None:raise error
            raise RuntimeError('nonlinear:grid: 原生调度计算失败')
        return output
    # 维护数值状态、依赖或计算接口。
    def begin(self):self.lib.grid_begin(self.handle)
    # 维护数值状态、依赖或计算接口。
    def rollback(self):self.lib.grid_rollback(self.handle)
    # 维护数值状态、依赖或计算接口。
    def __del__(self):
        if getattr(self,'handle',None):self.lib.grid_delete(self.handle);self.handle=None
    # 维护数值状态、依赖或计算接口。
    def stats(self):
        """返回累计原生工作量；有效位分支及显式目标请求按实际尝试计数，回退不减计数。"""
        result=(ct.c_uint64*2)();self.lib.grid_stats(self.handle,ct.addressof(result))
        detail=(ct.c_uint64*5)();self.lib.grid_stats_detail(self.handle,ct.addressof(detail))
        return dict(scalar_values_calculated=int(result[0]),calculator_batches=int(result[1]),
            validity_checks=int(detail[0]),invalidated_points=int(detail[1]),requested_outputs=int(detail[2]),
            checkpoint_points=int(detail[3]),direct_edges=int(detail[4]))
