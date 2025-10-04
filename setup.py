import numpy as np
from setuptools import setup, find_packages # 定义一个Python包的配置信息（比如名字、版本、描述等）
from distutils.extension import Extension # 用于定义C/C++扩展模块的配置
from Cython.Build import cythonize # 通过Cython将一个.pyx文件（Cython源码）编译成C扩展模块，提升性能。
# Cython可以让Python代码编译成C扩展，提高计算效率，尤其是在需要大量数值计算或循环的场景中非常有用。

# 定义函数获取numpy头文件目录，不同numpy版本方法名称可能不同，做兼容处理。
def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'reid.evaluation.rank_cylib.rank_cy', # Cython扩展模块的名称
        ['reid/evaluation/rank_cylib/rank_cy.pyx'], # Cython源码文件路径
        include_dirs=[numpy_include()], # 包含numpy头文件目录，确保Cython可以找到numpy的C API
    )
]
__version__ = '1.0.0' # 定义当前包的版本号

setup(
    name='ADSR', # 包的名称
    version='1.0.0', # 包的版本号
    description='DAmplitude-aware Domain Style Replay for Lifelong Person Re-identification',
    author='Long Chen', # 作者名称
    license='MIT, following Zhicheng Sun', # 许可证类型，遵循MIT协议
    packages=find_packages(), # 自动查找当前目录下的所有Python包
    keywords=['Person Re-Identification', 'Lifelong Learning', 'Computer Vision'], # 关键词列表，用于描述包的功能
    ext_modules=cythonize(ext_modules) # 使用Cython编译扩展模块
)