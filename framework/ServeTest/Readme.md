# 简介
本工具用来测试FD服务，主要功能是规范统一测试入口，减少重复代码，提高测试效率。统一封装http请求服务，llm请求模板，结果校验和后处理模块。

# 工具使用
**由于暂不支持入库逻辑，所以采用本地存储的方式进行对比校验，在新增测试时需要执行一遍基线数据获取**
+ 基线获取：
```python lanucher.py --request_template TOKEN_LOGPROB --url http://localhost:8000/v1/chat/completions  --case ./cases/demo.yaml  --concurrency 1 --name demo --exe logprob --baseline```

+ 测试执行（去掉baseline参数）：
```python lanucher.py --request_template TOKEN_LOGPROB --url http://localhost:8000/v1/chat/completions  --case ./cases/demo.yaml  --concurrency 1 --name demo --exe logprob```

## 参数说明

🎯 参数说明

--url（必填）

类型：str
示例：http://localhost:8000/v1/chat/completions
FastDeploy 服务 URL，必须以 http:// 或 https:// 开头。

⸻

--name（必填）

类型：str
示例：test_task_001
任务名称，用于记录、入库或区分不同测试任务。

⸻

--case（必填） 

类型：str
示例：./cases/test_case1.yaml
测试用例文件路径，必须为 YAML 格式，包含全局字段和测试用例列表。

⸻

--executor, -exe（必填）

类型：str
示例：logprob
执行器类型，需在代码的注册路由中存在，决定使用哪种用例处理逻辑。

⸻

--request_template, -rt（必填）

类型：str
示例：TOKEN_LOGPROB
请求模板名，对应 request_template.py 中定义的某个变量，如 TOKEN_LOGPROB。

⸻

--timeout

类型：int，默认值：60
请求超时时间（单位秒），每个请求的最大等待时长。

⸻

--concurrency

类型：int，默认值：4
并发请求数，仅在非 baseline 模式下生效。建议不超过 CPU/GPU 数。

⸻

--baseline

类型：bool（Flag 参数），默认值：False
是否启用基线模式。启用后，程序将把当前请求结果保存为基准输出；否则会与已有 baseline 做比较。

⸻

✅ 参数校验说明：
+ --url 必须以 http:// 或 https:// 开头，否则程序报错退出；
+ --case 指定的 YAML 文件路径必须存在，否则程序报错退出；
+ 所有标记为「必填」的参数未提供时将直接中断执行。


# 开发指南

程序核心逻辑是 ，根据模板和case构造http请求，执行请求，根据executor选择解析和比对逻辑，最后执行入库逻辑（目前是采用文件比对 ，后续统一修改）。

+ lanucher.py  入口文件，用来进行执行参数校验和主流程调度
+ utils.py 工具函数，http请求封装
+ case_loader.py 测试用例加载器，yaml解析, yaml内容格式参考demo
+ logger 日志分流模块
+ core 核心逻辑模块，包括结果解析，结果比对，入库逻辑等，不同的测试需要自己继承ResponseHandler
+ config 配置文件，包括模板路径等
+ cases 测试用例文件夹
