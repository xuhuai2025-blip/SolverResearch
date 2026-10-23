"""为附带 ctypes 原生库的安装包生成平台标记。"""
from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformDistribution(Distribution):
    """安装包包含与平台相关的原生调度器。"""
    def has_ext_modules(self):
        """禁止将动态库错误标记为跨平台纯 Python 包。"""
        return True


class CtypesWheel(bdist_wheel):
    """原生库使用 C ABI，不依赖 CPython 扩展 ABI。"""
    def get_tag(self):
        """保留平台标签，并使用 Python 3 通用解释器标签。"""
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform


setup(distclass=PlatformDistribution, cmdclass={'bdist_wheel':CtypesWheel})
