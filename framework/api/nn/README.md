
# 网络层测试套件
## 依赖安装
* pip install pytest

## 执行命令
**命令默认会选择test_开头的文件进行执行**
* pytest --disable-warnings . 执行全部
* pytest --disable-warnings -sv . 执行全部并打印其中print输出，用户debug=True时候的调试
* pytest . --html=report.html 生成Html测试报告，名字是report.html
* pytest --disable-warnings -sv xx.py::test_xx 执行xx.py下的test_xx测试用例

## 规范
每个API对应一个文件

## 使用方式
**参考test_example.py 或者 test_roll.py**
* 需要编写一个类继承APICase类，并且重写hook方法，hook方法主要需要包含self.types定义，同时也可选择性改写基类功能配置参数。
* docstring 写传入的参数列表，方便直观看到传入参数
* 实例化对象 obj。
* 编写case，指定输入输出，调用obj.run即可。
* 通用测试，例如check dtype调用 obj.base即可。

## 基本用法
* obj.run()执行专用case。
* obj.base()执行dtype等基础检测Case。
* obj.exception()执行异常检测Case，异常检测使用pytest.raise实现。默认检测c++异常，采用字符串匹配，异常类型是字符串。如果需要检测python异常，异常类型就是python异常类型，mode参数设置成为"python"

## 全局参数用法
* self.no_grad_var (tuple/set/list) 用来设置tensor类型的param反向截断，如果不设置，可能在反向check出现keyerror。
* self.debug=True 用于编写case的时候调试用，会将全部的log都打印出来，包括错误异常。默认值：False
* self.static=True和self.dygraph=True用于指定跑该类型网络，默认均为True，设置False为不跑该类型网络。
* self.enable_backward=False用来截断反向网络，主要是调试前向，以及某些无反向API的测试，默认为True。

## 特殊情况参数设置
比如在某些情况，dtype支持除float32，float64以外的类型，同时该api存在反向，正常执行因为int，bool类型不存在反向会报错，那么需要单独重新定义参数，有两种方法实现。第一种：在指定case里单独指定参数，比如obj.types=[]，同时执行完case之后需要把参数还原回去，防止下个case参数异常。第二种：单独定义指定相关Case的对象obj2，利用obj2进行一些特殊行为的Case测试。

## 不兼容情况
需要单独编写case，不依赖框架，具体写法可以参考test_meshgrid1

## FAQ
目前功能不完善，有bug联系dddivano@outlook.com。
