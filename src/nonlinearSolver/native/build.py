"""构建独立模块的本机原生调度库，保持各平台严格浮点设置。"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,tempfile
from nonlinearSolver.runtime.nativeLibrary import hostTarget,builtinPath,validateLibraryPath


# 维护数值状态、依赖或计算接口。
def compileCommand(target,source,output,compiler):
    """仅启用优化和平台代码生成，不启用 fast-math 或浮点融合。"""
    if target['platform']=='Windows':
        return [str(compiler),'/nologo','/utf-8','/O2','/std:c++17','/EHsc','/LD','/fp:strict',str(source),
                '/Fe:'+str(output),'/Fo:'+str(output.with_suffix('.obj')),'/link','/IMPLIB:'+str(output.with_suffix('.lib'))]
    return [str(compiler),'--sdk','macosx','clang++','-std=c++17','-O3','-dynamiclib','-arch','arm64',
            '-mmacosx-version-min=12.0','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden',str(source),
            '-Wl,-install_name,@rpath/'+output.name,'-o',str(output)]



# 维护数值状态、依赖或计算接口。
def windowsEnvironment():
    """优先使用已有开发命令行，否则通过 vswhere 定位已安装的 C++ 工具。"""
    environment=os.environ.copy();compiler=shutil.which('cl')
    if compiler:return compiler,environment
    where=Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    if not where.is_file():raise ValueError('请安装 Visual Studio C++ Build Tools 后再构建调度库')
    location=subprocess.check_output([str(where),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
    setup=Path(location)/'VC/Auxiliary/Build/vcvars64.bat'
    if not location or not setup.is_file():raise ValueError('未找到 MSVC x64 编译环境')
    # 这里只执行已定位的开发环境脚本；用户源文件和输出路径始终按独立参数传给 cl。
    command='cmd /d /s /c "call "'+str(setup)+'" >nul && set"'
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise ValueError('无法初始化 MSVC x64 编译环境')
    for line in result.stdout.splitlines():
        if '=' in line and not line.startswith('='):
            key,value=line.split('=',1);environment[key]=value
    compiler=shutil.which('cl',path=environment.get('Path',environment.get('PATH','')))
    if not compiler:raise ValueError('MSVC 环境未提供 cl 编译器')
    return compiler,environment


# 维护数值状态、依赖或计算接口。
def build(output=None):
    """先在目标目录内临时构建，成功后原子替换动态库；失败保留原库。"""
    target=hostTarget();output=Path(output).resolve() if output else builtinPath(target)
    if output.suffix.lower()!=target['suffix']:raise ValueError('输出文件扩展名必须为 '+target['suffix'])
    output.parent.mkdir(parents=True,exist_ok=True);environment=os.environ.copy()
    if target['platform']=='Windows':compiler,environment=windowsEnvironment()
    else:
        compiler=shutil.which('xcrun')
        if not compiler:raise ValueError('请安装 Xcode Command Line Tools 后构建 arm64 调度库')
    with tempfile.TemporaryDirectory(prefix='.scheduler-build-',dir=output.parent) as directory:
        temporary=Path(directory)/output.name
        command=compileCommand(target,Path(__file__).with_name('grid_runtime.cpp').resolve(),temporary,compiler)
        subprocess.run(command,env=environment,check=True)
        if target['platform']=='macOS':
            subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(temporary)],check=True)
        validateLibraryPath(temporary,target)
        try:os.replace(temporary,output)
        except PermissionError as exc:raise ValueError('调度库正在使用，请关闭相关计算进程后重新构建') from exc
    print('原生调度库构建完成：'+str(output))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='构建当前平台的原生调度库')
    parser.add_argument('--output',help='可选的动态库输出路径');args=parser.parse_args()
    try:build(args.output)
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print('调度库构建失败：'+str(exc),file=sys.stderr);raise SystemExit(1)
