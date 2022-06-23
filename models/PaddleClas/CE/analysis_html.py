"""解析html结果"""

import requests
from bs4 import BeautifulSoup
import numpy as np

# hang=3 #gan
hang = 7  # clas
lie = 9
lie_index = 5
b = []

url_all = [
    "https://xly.bce.baidu.com/ipipe/ipipe-report/report/15746602/result/reportUrl.html",
    "https://xly.bce.baidu.com/ipipe/ipipe-report/report/15777850/result/reportUrl.html",
]  # release
with open("clas_release", "w", encoding="utf-8") as f:

    # url_all = [
    #     "https://xly.bce.baidu.com/ipipe/ipipe-report/report/15715846/result/reportUrl.html",
    #     "https://xly.bce.baidu.com/ipipe/ipipe-report/report/15779389/result/reportUrl.html",
    # ] #develop
    # with open("clas_develop", "w", encoding="utf-8") as f:

    for i, _ in enumerate(url_all):
        # for i in range(len(url_all)): #code style error
        url = url_all[i]
        resp = requests.get(url)
        # print(resp.content)
        # 可以设置响应结果的编码
        resp.encoding = "utf-8"
        # 使用 bs4 解析数据
        # 1.将页面源代码交给 BeautifulSoup 进行处理生成 bs 对象
        bs = BeautifulSoup(resp.text, features="html.parser")
        # bs = BeautifulSoup(open("gan-p0-dev.html"),features="html.parser")#html.parser是解析器，也可是lxml 需要单独装包

        # 2.从bs对象中查找数据  find(标签， 属性=值)方法(找第一个) 和 find_all(标签， 属性=值)方法(找全部)
        table = bs.find("table")
        # id 也可以放在 attrs 中  attrs={"id": "myTh"}   class为python关键字所以用 class_作为 key
        thList = table.find_all("td", attrs={}, class_=[])
        # a=[]
        # thList = table.find_all("th", attrs={}, class_=[])
        # for t in thList:
        #     # 使用 get方法  获取属性值
        #     # print(t.get("text"))  #None
        #     # get_text() 获取标签里的内容 eg: <p>just for example </p>  返回 just for example
        #     # print('####',t.get_text())
        #     a.append(t.get_text())
        # b.append(a) #表头先不要
        i = 0
        k = 0
        tmp = []
        for t in thList:
            # 使用 get方法  获取属性值
            # print(t.get("text"))  #None
            # get_text() 获取标签里的内容 eg: <p>just for example </p>  返回 just for example
            # print(t.get_text())
            i += 1
            # print('###i',i)

            if k >= hang:  # 3是一个模型有的参数数量
                f.write("\n")
                b.append(tmp)
                tmp = []
                k = 0
            if i == lie_index or (i == 1 and k == 0):  # 5是第五列
                # if i<=8:
                # print('###',t.get_text())
                tmp.append(t.get_text())
                f.write(t.get_text() + ",")
                # print('####tmp',tmp)
            elif i >= lie:  # 9是一共有9列
                # print('###',t.get_text())
                # input()
                i = 0
                k += 1
                # print('@@@@b',b)
                # input()
            else:
                pass
        f.write("\n")
    # f.write("\n")
f.close()
print("####b", b)
