"""Python BDF 刚性 ODE 和 index-1 DAE 的统一积分接口。"""
import time
import numpy as np
from scipy_dae.integrate import solve_dae
from scipy_dae.integrate._dae.bdf import BDFDAE
from scipy_dae.integrate._dae.common import consistent_initial_conditions

from dasslSolver.problem import asProblem, vector
from dasslSolver.evaluator import Evaluator


class IntegrationResult(dict):
    """允许以属性或字段名称读取积分结果。"""
    def __getattr__(self,name):
        """缺失字段按普通属性错误报告。"""
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class _DenseOutput:
    """统一后端单变量标量查询的输出形状。"""
    def __init__(self,inner,size):
        """保存插值器，不修改插值多项式。"""
        self.inner=inner
        self.size=size

    def __call__(self,times):
        """标量查询返回状态向量，向量查询返回状态乘时间的矩阵。"""
        query=np.asarray(times)
        state,derivative=self.inner(times)
        shape=(self.size,) if query.ndim==0 else (self.size,query.size)
        return np.asarray(state).reshape(shape),np.asarray(derivative).reshape(shape)


class TransientSolver:
    """保留通用瞬态接口；积分内核由固定版本 scipy-dae 提供。"""
    def __init__(self, options=None):
        """拒绝未知选项，逐项验证所有会影响积分的设置。"""
        defaults = dict(RelTol=1e-6,AbsTol=1e-8,MaxStep=None,InitialStep=None,
            MaxOrder=5,BDF=True,Stats=False,OutputTimes=None,
            ComputeConsistentInitialConditions=True,MaxSteps=100000)
        options = {} if options is None else dict(options)
        if set(options)-set(defaults):
            raise ValueError('dassl:UnknownOption: 未知选项 '+', '.join(sorted(set(options)-set(defaults))))
        self.Options = dict(defaults,**options)
        for name in ('RelTol',):
            value = self.Options[name]
            if not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError('dassl:Options: '+name+'必须为有限正标量')
        for name in ('MaxStep','InitialStep'):
            value = self.Options[name]
            if value is not None and (not np.isscalar(value) or not np.isfinite(value) or value<=0):
                raise ValueError('dassl:Options: '+name+'必须为有限正标量')
        if type(self.Options['MaxOrder']) is not int or not 1<=self.Options['MaxOrder']<=5:
            raise ValueError('dassl:MaxOrder: 最大阶数必须是 1 至 5 的整数')
        if type(self.Options['MaxSteps']) is not int or self.Options['MaxSteps']<1:
            raise ValueError('dassl:MaxSteps: 最大步数必须为正整数')
        for name in ('BDF','Stats','ComputeConsistentInitialConditions'):
            if not isinstance(self.Options[name],(bool,np.bool_)):
                raise ValueError('dassl:Options: '+name+'必须为布尔值')

    def _mask(self,value,size):
        """初值固定掩码使用与状态等长的布尔向量。"""
        if value is None:
            return np.zeros(size,dtype=bool)
        value = np.asarray(value)
        if value.ndim==0 and size==1:
            value=value.reshape(1)
        if value.shape!=(size,) or value.dtype.kind!='b':
            raise ValueError('dassl:InitialMask: 初值固定掩码必须为与状态等长的布尔向量')
        return value.copy()

    def _events(self,problem,evaluator,t0,y0,yp0):
        """支持原接口的事件值、终止标记和方向三元组。"""
        if problem.Events is None:
            return None
        def values(t,y,yp):
            raw = problem.Events(t,y.copy(),yp.copy()) if problem.form=='residual' else problem.Events(t,y.copy())
            if not isinstance(raw,(tuple,list)) or len(raw)!=3:
                raise ValueError('dassl:Events: 事件必须返回 value、terminal、direction 三元组')
            value = vector(raw[0],'事件值')
            terminal = np.atleast_1d(raw[1])
            direction = np.atleast_1d(raw[2])
            if terminal.shape!=value.shape or direction.shape!=value.shape or not np.isfinite(direction).all():
                raise ValueError('dassl:Events: 事件值、终止标记和方向的长度必须一致')
            if not np.all((terminal==0)|(terminal==1)):
                raise ValueError('dassl:Events: 事件终止标记必须为布尔值或 0/1')
            return value,terminal.astype(bool),direction
        initial,terminal,direction = values(t0,y0,yp0)
        events = []
        for index in range(initial.size):
            def event(t,y,yp,index=index):
                current,_,_ = values(t,y,yp)
                if current.size!=initial.size:
                    raise ValueError('dassl:Events: 事件数量不能在积分过程中变化')
                return current[index]
            event.terminal=bool(terminal[index])
            event.direction=float(direction[index])
            events.append(event)
        return events

    def solve(self,problem,timeSpan,initialState,initialDerivative=None):
        """积分到终点、终止事件或输出回调停止，并记录独立终态残差。"""
        started = time.perf_counter()
        problem = asProblem(problem)
        span = vector(timeSpan,'时间区间')
        if span.size<2 or np.any(np.diff(span)<=0):
            raise ValueError('dassl:TimeSpan: 时间区间必须有限且严格递增')
        requested = span.copy() if self.Options['OutputTimes'] is None else vector(self.Options['OutputTimes'],'输出时间')
        if np.any(np.diff(requested)<0) or requested[0]<span[0] or requested[-1]>span[-1]:
            raise ValueError('dassl:OutputTimes: 输出时间须排序且位于积分区间内')
        y0 = vector(initialState,'初始状态')
        inputY0 = y0.copy()
        inputYP0 = None if initialDerivative is None else vector(initialDerivative,'初始导数',y0.size)
        yp0 = np.zeros_like(y0) if inputYP0 is None else inputYP0.copy()
        atol = vector(self.Options['AbsTol'],'绝对容差')
        if atol.size not in (1,y0.size) or np.any(atol<=0):
            raise ValueError('dassl:AbsTol: 绝对容差必须为正标量或与状态等长的正向量')
        atol = float(atol[0]) if atol.size==1 else atol
        evaluator = Evaluator(problem,y0.size)
        evaluator.validate(span[0],y0)
        patterns = evaluator.patterns(span[0],y0)
        jac = None if problem.Jacobian is None else evaluator.jacobian
        if jac is not None:
            jac(span[0],y0,yp0)
        nonnegative = np.asarray([],int) if problem.NonNegative is None else np.atleast_1d(problem.NonNegative)
        if nonnegative.dtype.kind not in 'iu' or np.any((nonnegative<0)|(nonnegative>=y0.size)):
            raise ValueError('dassl:NonNegative: 非负分量必须为有效的零起始整数索引')
        if nonnegative.size and problem.form!='explicit':
            raise ValueError('dassl:NonNegativeUnsupported: 非负分量检查仅支持显式 ODE')
        fixedY = self._mask(problem.FixedY0,y0.size)
        fixedYP = self._mask(problem.FixedYP0,y0.size)
        initializationStarted = time.perf_counter()
        # 显式 ODE 固定输入状态，初始导数由真实右端给出。
        if problem.form=='explicit':
            yp0 = evaluator.rhs(span[0],y0)
        elif self.Options['ComputeConsistentInitialConditions']:
            # 后端接口使用固定分量的整数索引，布尔掩码先显式转换。
            y0,yp0,_ = consistent_initial_conditions(evaluator.residual,float(span[0]),y0.copy(),yp0.copy(),
                jac=None if jac is None else evaluator.denseJacobian,
                fixed_y0=np.flatnonzero(fixedY),fixed_yp0=np.flatnonzero(fixedYP),
                rtol=float(self.Options['RelTol']),atol=atol)
        # 不计算一致初值时也必须核验输入，不能把不一致的起点当作有效解。
        initialResidual = evaluator.residual(span[0],y0,yp0)
        threshold = 10*np.max(np.atleast_1d(atol))+10*self.Options['RelTol']*max(1.,np.linalg.norm(yp0,np.inf))
        if np.linalg.norm(initialResidual,np.inf)>threshold:
            raise ValueError('dassl:InconsistentInitialConditions: 初值不满足残差方程')
        initializationSeconds = time.perf_counter()-initializationStarted
        events = self._events(problem,evaluator,span[0],y0,yp0)
        options = self.Options
        callback = problem.OutputFcn
        callbackStopped = [False]
        acceptedSteps = []
        orders = []

        def notify(t,y,flag):
            """输出回调使用状态副本，停止请求只在已接受步生效。"""
            if callback is None:
                return False
            value = callback(float(t),y.copy(),flag)
            if value is None:
                return False
            if np.size(value)!=1:
                raise ValueError('dassl:OutputFcn: 输出回调必须返回单个布尔停止标记或 None')
            return bool(value)

        if notify(span[0],y0,'init'):
            raise ValueError('dassl:OutputFcn: init 阶段不能请求停止，请在 step 阶段返回停止标记')

        class ObservedBDF(BDFDAE):
            """在后端接受步上记录阶数、执行回调并检查步数上限。"""
            def __init__(self,*args,**kwargs):
                """BDF=True 使用纯 BDF；否则保留后端 NDF 修正。"""
                super().__init__(*args,**kwargs)
                if options['BDF']:
                    self.alpha=self.gamma.copy()
                    self.error_const=1/np.arange(1,self.max_order+2)

            def dense_output(self):
                """事件寻根和结果查询均使用相同的向量形状合同。"""
                return _DenseOutput(super().dense_output(),self.n)

            def step(self):
                """不修改状态，只对后端接受步作检查及进度通知。"""
                message=super().step()
                if self.status!='failed':
                    acceptedSteps.append(float(self.t))
                    orders.append(int(self.order))
                    if len(acceptedSteps)>options['MaxSteps']:
                        raise RuntimeError('dassl:MaxSteps: 超过允许的积分步数')
                    if nonnegative.size and np.any(self.y[nonnegative]<0):
                        raise ValueError('dassl:NonNegative: 已接受状态违反非负约束，不进行静默裁剪')
                    if notify(self.t,self.y,'step'):
                        callbackStopped[0]=True
                        self.status='finished'
                return message

        integrationStarted=time.perf_counter()
        solution=solve_dae(evaluator.residual,(float(span[0]),float(span[-1])),y0,yp0,
            method=ObservedBDF,rtol=float(options['RelTol']),atol=atol,
            max_step=np.inf if options['MaxStep'] is None else float(options['MaxStep']),
            first_step=options['InitialStep'],max_order=options['MaxOrder'],
            jac=jac,jac_sparsity=patterns,events=events,dense_output=True)
        integrationSeconds=time.perf_counter()-integrationStarted
        if not solution.success:
            raise RuntimeError('dassl:IntegrationFailed: 积分未成功，后端信息：'+str(solution.message))
        actualEnd=float(solution.t[-1])
        slack=10*np.finfo(float).eps*max(1.,abs(actualEnd))
        query=requested[requested<=actualEnd+slack].copy()
        if query.size==0 or query[-1]<actualEnd-slack:
            query=np.r_[query,actualEnd]
        else:
            query[-1]=actualEnd

        def evaluate(times):
            """连续输出仅在实际积分区间内有效，不允许外推。"""
            t=np.asarray(times,float)
            if t.ndim>1 or not np.isfinite(t).all() or np.any((t<span[0])|(t>actualEnd)):
                raise ValueError('dassl:EvaluateTime: 查询时间必须位于实际积分区间内')
            if t.ndim==1 and t.size==0:
                return np.empty((y0.size,0)),np.empty((y0.size,0))
            return solution.sol(t)

        state,derivative=evaluate(query)
        finalResidual=evaluator.residual(actualEnd,state[:,-1],derivative[:,-1])
        actualY0,actualYP0=evaluate(span[0])
        notify(actualEnd,state[:,-1],'done')
        totalSeconds=time.perf_counter()-started
        result=IntegrationResult(Solution=solution,Time=query,State=state,Derivative=derivative,
            InputInitialState=inputY0,InputInitialDerivative=inputYP0,
            InitialState=actualY0,InitialDerivative=actualYP0,
            FinalState=state[:,-1],FinalDerivative=derivative[:,-1],
            FinalResidual=finalResidual,FinalResidualInf=float(np.linalg.norm(finalResidual,np.inf)),
            InitialResidualInf=float(np.linalg.norm(initialResidual,np.inf)),
            ProblemForm=problem.form,Backend='scipy-dae-BDF',
            EndReason='event' if solution.status==1 else 'output_callback' if callbackStopped[0] else 'final_time',
            InternalMeshPointCount=solution.t.size,AcceptedStepTimes=np.asarray(acceptedSteps),
            AcceptedStepOrders=np.asarray(orders),FunctionEvaluations=solution.nfev,
            JacobianEvaluations=solution.njev,LinearFactorizations=solution.nlu,
            ConsistentInitializationElapsedSeconds=initializationSeconds,
            IntegrationElapsedSeconds=integrationSeconds,TotalElapsedSeconds=totalSeconds,
            Options=dict(options),Evaluate=evaluate)
        if options['Stats']:
            print(f'BDF 积分完成：步数={len(acceptedSteps)}，终态残差无穷范数={result.FinalResidualInf:.6e}')
        return result


def solve(problem,timeSpan,initialState,options=None,initialDerivative=None):
    """以函数形式调用同一瞬态积分器。"""
    return TransientSolver(options).solve(problem,timeSpan,initialState,initialDerivative)
