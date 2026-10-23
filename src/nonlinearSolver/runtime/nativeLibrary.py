"""独立原生调度库定位：Windows x64 与 macOS arm64。"""
from pathlib import Path
import os,platform,struct,sys

ENVIRONMENT_KEY='NONLINEAR_SOLVER_LIBRARY'
_configured=None


# 维护数值状态、依赖或计算接口。
def hostTarget(system=None,machine=None,bits=None):
    """以当前 Python 进程架构选择动态库，拒绝将其他架构的库混入进程。"""
    system=sys.platform if system is None else system
    machine=platform.machine().lower() if machine is None else machine.lower()
    bits=struct.calcsize('P')*8 if bits is None else bits
    if bits==64 and system=='win32' and machine in ('amd64','x86_64'):
        return dict(platform='Windows',architecture='x86_64',suffix='.dll',filename='nonlinear_runtime.dll')
    if bits==64 and system=='darwin' and machine in ('arm64','aarch64'):
        return dict(platform='macOS',architecture='arm64',suffix='.dylib',filename='libnonlinear_runtime.dylib')
    raise ValueError('原生求解模块需要 Windows x64 或 macOS arm64 的标准 64 位 Python')


# 维护数值状态、依赖或计算接口。
def builtinPath(target=None):
    target=hostTarget() if target is None else target
    return Path(__file__).resolve().parents[1]/'native'/target['filename']


# 维护数值状态、依赖或计算接口。
def configureLibrary(path=None):
    """只影响后续创建的调度器；已有调度器继续持有自己的库句柄。"""
    global _configured
    _configured=Path(path).expanduser().resolve() if path else None


# 维护数值状态、依赖或计算接口。
def selectedPath():
    """后台作业的显式环境选择优先，其次是本进程配置，最后采用平台内置库。"""
    value=os.environ.get(ENVIRONMENT_KEY)
    return Path(value).expanduser().resolve() if value else _configured or builtinPath()


# 维护数值状态、依赖或计算接口。
def validateLibraryPath(path,target=None):
    """在加载前核对 PE/Mach-O 格式及架构；不执行所选文件。"""
    target=hostTarget() if target is None else target;path=Path(path).expanduser().resolve()
    if path.suffix.lower()!=target['suffix']:raise ValueError('当前平台请选择 '+target['suffix']+' 调度库')
    if not path.is_file():raise ValueError('找不到调度库：'+str(path)+'；请运行 python -m nonlinearSolver.native.build 构建')
    size=path.stat().st_size
    with path.open('rb') as stream:
        header=stream.read(64)
        if target['platform']=='Windows':
            if len(header)<64 or header[:2]!=b'MZ':raise ValueError('所选文件不是 Windows 动态库')
            offset=struct.unpack_from('<I',header,60)[0]
            if offset>size-26:raise ValueError('Windows 动态库文件头不完整')
            stream.seek(offset);pe=stream.read(26)
            if pe[:4]!=b'PE\0\0' or struct.unpack_from('<H',pe,4)[0]!=0x8664 or struct.unpack_from('<H',pe,24)[0]!=0x20b:
                raise ValueError('调度库必须为 Windows x64 架构')
            if not struct.unpack_from('<H',pe,22)[0]&0x2000:raise ValueError('所选 PE 文件不是 DLL')
        else:
            # 同时支持单 arm64 和包含 arm64 切片的 Universal 动态库。
            def arm64Dylib(offset=0,length=None):
                if offset<0 or offset>size-32 or length is not None and (length<32 or offset+length>size):return False
                stream.seek(offset);data=stream.read(32)
                return data[:4]==b'\xcf\xfa\xed\xfe' and struct.unpack_from('<I',data,4)[0]==0x0100000c and struct.unpack_from('<I',data,12)[0]==6
            if not arm64Dylib():
                magic=header[:4];formats={b'\xca\xfe\xba\xbe':('>',False),b'\xca\xfe\xba\xbf':('>',True),b'\xbe\xba\xfe\xca':('<',False),b'\xbf\xba\xfe\xca':('<',True)}
                if magic not in formats:raise ValueError('调度库必须为 macOS arm64 动态库')
                if len(header)<8:raise ValueError('macOS 动态库文件头不完整')
                endian,wide=formats[magic];count=struct.unpack_from(endian+'I',header,4)[0]
                if not 1<=count<=32:raise ValueError('Universal 动态库切片表无效')
                stream.seek(8);table=stream.read(count*(32 if wide else 20));found=False
                for index in range(count):
                    start=index*(32 if wide else 20)
                    if len(table)<start+(32 if wide else 20):raise ValueError('Universal 动态库文件头不完整')
                    cpu=struct.unpack_from(endian+'I',table,start)[0]
                    offset,length=struct.unpack_from(endian+('QQ' if wide else 'II'),table,start+8)
                    if cpu==0x0100000c and arm64Dylib(offset,length):found=True
                if not found:raise ValueError('Universal 动态库没有有效的 arm64 动态库切片')
    return path
