# coding: utf-8
 
from setuptools import setup, find_packages
 
setup(
    name='Test',  # 项目名称，pip show 包名 中的包名
    version='1.0.0',
    packages=find_packages(), # 包含所有的py文件
    include_package_data=True, # 将数据文件也打包
    zip_safe=True,
    entry_points={
        'console_scripts': [ # 命令的入口
            'test_start=Test.command:start',  # test_start命令对应的入口函数为command.py下的start函数
            'test_init=Test.command:init' # test_init命令对应的函数为command.py下的init函数
        ]
    }
)
