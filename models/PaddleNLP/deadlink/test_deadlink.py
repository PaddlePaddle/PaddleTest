# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
get and check link by request url
"""

import os
import pickle
from bs4 import BeautifulSoup
import requests
import copy
import re
import datetime
import time
import json
import sys

import run_test_doc as lib_wy28
import conf
from urllib.parse import urlparse
requests.packages.urllib3.disable_warnings()

from args import parse_args

EXCEL_TITLE = ['抓取结果', 'link', '链接名',
               'page', '页面名', '版本',
               'owner', 'key', '一级目录', '二级目录', '三级目录', '四级目录',
               '五级目录', 'link_host']
DB_COL = ['status_code', 'link', 'link_name',
          'page', 'page_name', 'version',
          'owner', 'link_key', 'path_1', 'path_2', 'path_3', 'path_4',
          'path_5', 'link_host', 'insert_db_time']

LIST_WHITE = lib_wy28.load_white()

def spider_rst_links():
    """
    https://www.cnblogs.com/ls1519/p/12635091.html
    Returns:

    """


def filter_url(page, basename, block_filter):
    """
    拼装有效的url
    Returns:

    """
    if basename in ['http://localhost:8888', 'http://0.0.0.0:8080',
                    'chrome://tracing/', '', '#']:
        url = ''
    elif block_filter == {'class': 'wy-side-scroll'}:
        url = '%s%s' % (conf.PADDLE_HOST, basename)
    elif '#' in page:
        # 防止出现链里有两个锚点#
        url = ''
    elif basename.startswith('http') or basename.startswith('ftp://'):
        url = basename
    elif basename.startswith('#'):
        url = '%s%s' % (page, basename)
    else:
        url = '%s/%s' % (os.path.dirname(page), basename)
    url = url.replace('_pr/', '/')
    return url


def deduplicate_links(links):
    seen = {}
    for link in links:
        url = link['url']
        # 如果已经出现过该 url，跳过 direct link，但保留命名链接
        if url in seen:
            # 优先保留非 direct link 的版本
            if link['name'] != 'direct link' and seen[url]['name'] == 'direct link':
                seen[url] = link
        else:
            seen[url] = link
    return list(seen.values())


def spider_md_links(file_md, repo_name, version):
    """
    获取 md 文件中的所有链接，包括 Markdown 链接和裸露链接/下载命令
    Returns:
    [超链接名](超链接地址 "超链接title")
    string = 'abe(ac)ad)'
    p1 = re.compile(r'[(](.*?)[)]', re.S) #最小匹配
    p2 = re.compile(r'[(](.*)[)]', re.S) #贪婪匹配
    ['ac']
    ['ac)ad']
    1.正则匹配串前加了r就是为了使得里面的特殊符号不用写反斜杠了
    2.[ ]具有去特殊符号的作用,也就是说[(]里的(只是平凡的括号
    3.正则匹配串里的()是为了提取整个正则串中符合括号里的正则的内容
    4.是为了表示除了换行符的任一字符。*克林闭包，出现0次或无限次。
    5.加了？是最小匹配，不加是贪婪匹配。
    6.re.S是为了让.表示除了换行符的任一字符。
    """
    links = []
    repo_path = file_md.split(repo_name)[-1]
    page = 'https://github.com/PaddlePaddle/%s/tree/%s%s' % (repo_name, version, repo_path)

    with open(file_md, 'r') as fr:
        str = fr.read()
    # 1. 提取 Markdown 格式的链接
    pattern = r'[[](.*?)[]][(](.*?)[)]'
    link_list = re.findall(pattern, str)
    for link in link_list:
        basename = link[1]
        url = filter_url(page, basename, None)
        if url != '':
            link_dict = {}
            link_dict['name'] = link[0]
            link_dict['url'] = url
            links.append(link_dict)
    # 2. 提取裸露的 http/https 链接（比如 tar.gz 下载链接）
    allowed_exts = ('.tar.gz', '.gz', '.zip', '.txt', '.md', '.tar', '.json', '.csv', '.bin', '.idx')
    raw_url_pattern = r'https?://[^\s)]+'
    all_links = re.findall(raw_url_pattern, str)
    filtered_links = [url for url in all_links if url.endswith(allowed_exts)]
    for link in filtered_links:
        link_dict = {}
        link_dict['name'] = 'direct link'
        link_dict['url'] = link
        links.append(link_dict)
    return deduplicate_links(links)


def spider_html_block_links(page, block_filter):
    """
    根据html块（某个div）获取某个页面里的所有链接
    Args:
        page:
        block_filter:
    Returns:

    """
    links = []
    # 抓取
    page = page.strip()
    try:
        # 设置TCP连接超时为3s，HTTP请求超时为7s
        r = requests.get(page, timeout=(3, 7), verify=False)
        status = r.status_code
        if status == 200:
            print('[url request success][%d]%s' % (status, page))
        else:
            print('[url request fail][%d]%s' % (status, page))
            # return links
            # exit(1)
    except Exception as err:
        # 发生错误时回滚
        print('[url request fail][%s]%s' % (str(err), page))
        # return links
        # exit(1)

    soup = BeautifulSoup(r.text, 'html5lib')
    try:
        for a in soup.find(attrs=block_filter).find_all('a'):
            try:
                basename = a['href']
                url = filter_url(page, basename, block_filter)
                # 去除锚点页
                if url != '' and '#' not in url:
                    link_dict = {}
                    link_dict['name'] = a.get_text()
                    link_dict['url'] = url
                    links.append(link_dict)
            except Exception as err:
                print('a without href[error:%s][page:%s]' % (a, page))
    except Exception as err:
        print('[page has no link][page:%s]' % page)
    return links


def get_md_links(lan_version, code_path, filename_sub, repo_name):
    """
    获取(含md文件的)文件夹里的md文件中所有link
    Returns:

    """
    pages_links = []
    files = lib_wy28.get_file_list_filter(code_path, '.md')
    for file in files:
        # '/Users/wangying28/Documents/paddle_github/FluidDoc/doc/fluid/
        # beginners_guide/basic_concept/dygraph/DyGraph.md'
        if "csrc/third_party" in file:
            # 跳过第三方库
            continue
        else:
            page_dict = {}
            page_dict['lan_version'] = lan_version
            page_dict['page'] = {'url': file, 'name': ''}
            page_dict['links'] = spider_md_links(file, repo_name, lan_version)
            pages_links.append(page_dict)

    file_pkl = '%s.pkl' % filename_sub
    # 清理环境
    if os.path.isfile(file_pkl):
        os.remove(file_pkl)
    # 存dict到pkl
    f = open(file_pkl, 'wb')
    pickle.dump(pages_links, f)
    f.close()


def get_doc_links(lan_version, filename_sub):
    """
    获取某个版本某个语言的所有文档链接
    Args:
        lan_version:
    Returns:

    """
    pages_links = []
    # 【所有的文档上、左导航链接】wy-side-scroll
    if lan_version.startswith('zh'):
        index = '%s/documentation/docs/%s/index_cn.html' % (conf.PADDLE_HOST, lan_version)
    elif lan_version.startswith('en'):
        index = '%s/documentation/docs/%s/index_en.html' % (conf.PADDLE_HOST, lan_version)
    # 文档的所有page
    doc_pages = spider_html_block_links(index, {'class': 'wy-side-scroll'})
    print('doc[%s]page num: %d' % (lan_version, len(doc_pages)))

    for i, page in enumerate(doc_pages):
        # 去除锚点页
        if '#' in page['url']:
            continue
        page_dict = {}
        page_dict['lan_version'] = lan_version
        page_dict['page'] = page
        # 文档主体内容:document
        center_links = []
        print('[%d-%d]' % (len(doc_pages), i))
        center_links = spider_html_block_links(page['url'], {'class': 'document'})
        page_dict['links'] = center_links
        # 【暂不用】右侧锚点:navigation-toc
        # center_links = spider_html_block_links(page['url'], {'class': 'navigation-toc'})
        pages_links.append(page_dict)

    file_pkl = '%s.pkl' % filename_sub
    # 清理环境
    if os.path.isfile(file_pkl):
        os.remove(file_pkl)
    # 存dict到pkl
    f = open(file_pkl, 'wb')
    pickle.dump(pages_links, f)
    f.close()
    return pages_links


def save_links(lan_version, filename_sub):
    """
    存dict到pkl，存txt，存Excel
    Args:
        pages_links:
        lan_version:
    Returns:

    """
    file_txt = '%s.txt' % filename_sub
    file_excel = '%s.xlsx' % filename_sub
    excel_table = [EXCEL_TITLE]
    # 获取链接
    file_pkl = '%s.pkl' % filename_sub
    if os.path.isfile(file_pkl):
        f = open(file_pkl, 'rb')
        pages_links = pickle.load(f)
        f.close()

    # 清理环境
    if os.path.isfile(file_txt):
        os.remove(file_txt)
    if os.path.isfile(file_excel):
        os.remove(file_excel)

    # 存txt
    result_txt = open(file_txt, 'a+')
    num = 0
    for page in pages_links:
        excel_line = ['', '', '']
        excel_line.append(page['page']['url'])
        excel_line.append(page['page']['name'])
        excel_line.append(lan_version)

        for link in page['links']:
            link_line = copy.deepcopy(excel_line)
            link_line[1] = link['url']
            link_line[2] = link['name']
            result_txt.write('\t'.join(link_line) + '\n')
            excel_table.append(link_line)
            num += 1
    result_txt.close()
    # 存excel
    lib_wy28.write_excel(file_excel, excel_table)
    print('doc[%s]link num: %d' % (lan_version, num))


def test_deadlink(source_file, filename_sub):
    """
    测试状态
    Args:
        source_file:
        filename_sub:

    Returns:

    """
    excel_table = [EXCEL_TITLE]
    # 清理环境
    file_txt = '%s.txt' % filename_sub
    file_excel = '%s.xlsx' % filename_sub
    if os.path.isfile(file_txt):
        os.remove(file_txt)
    if os.path.isfile(file_excel):
        os.remove(file_excel)

    link_list = lib_wy28.read_excel(source_file, '工作表1')
    # 存txt
    result_txt = open(file_txt, 'a+')
    for i, link in enumerate(link_list):
        # 去表头
        if i == 0:
            continue
        # 深拷贝，去None，方便join
        link_line = []
        for link_item in link:
            if link_item is None:
                link_item = ''
            link_line.append(str(link_item))

        url = link_line[1].strip()
        # jump white list
        if white_url(url):
            continue

        source = link_line[3]
        try:
            # 设置TCP连接超时为3s，HTTP请求超时为7s
            r = requests.get(url, timeout=(3, 7), verify=False)
            status = str(r.status_code)
            if status[0] not in ['2', '3']:
                print('[url request fail][%s]%s[source:%s]' % (status, url, source))
        except Exception as err:
            # 发生错误时回滚
            status = str(err)
            print('[url request fail][%s]%s[source:%s]' % (status, url, source))
        link_line[0] = status

        result_txt.write('\t'.join(link_line) + '\n')
        excel_table.append(link_line)
    result_txt.close()
    # 存excel
    lib_wy28.write_excel(file_excel, excel_table)
    return excel_table


def add_doc_owner(source_file, f_result_owner_sub, lan_version):
    """

    Returns:

    """
    excel_table = [EXCEL_TITLE]

    # 清理环境
    file_txt = '%s.txt' % f_result_owner_sub
    file_excel = '%s.xlsx' % f_result_owner_sub
    if os.path.isfile(file_txt):
        os.remove(file_txt)
    if os.path.isfile(file_excel):
        os.remove(file_excel)

    # # 获取负责人
    # doc_owner = lib_wy28.get_doc_owner(conf.OWNER_DOC_PATH, conf.OWNER_DOC_SHEET)
    # api_owner = lib_wy28.get_api_owner_new(conf.OWNER_API_PATH, conf.OWNER_API_SHEET)
    print(source_file)
    link_list = lib_wy28.read_excel(source_file, '工作表1')
    # 存txt
    result_txt = open(file_txt, 'a+')
    for i, link in enumerate(link_list):
        # 去表头
        if i == 0:
            continue
        # 深拷贝，去None，方便join
        link_line = []
        for link_item in link:
            if link_item is None:
                link_item = ''
            link_line.append(str(link_item))

        page = link_line[3].strip()
        # 层级关系
        path = page.split(lan_version)[-1].split('.')[0]
        key = path.replace('_cn', '').replace('_en', '')
        owner = ''
        # if key in doc_owner.keys():
        #     owner = doc_owner[key]['owner']
        # elif key in api_owner.keys() and api_owner[key]['owner'] is not None:
        #     owner = api_owner[key]['owner']
        link_line[6] = owner
        link_line[7] = key
        for i, level in enumerate(path.split('/')):
            # 有的目录超过5层
            if 0 < i < 6:
                link_line[i + 7] = level

        parsed_uri = urlparse(link_line[1])
        link_line[-1] = '%s://%s' % (parsed_uri.scheme, parsed_uri.netloc)

        result_txt.write('\t'.join(link_line) + '\n')
        excel_table.append(link_line)
    result_txt.close()
    # 存excel
    lib_wy28.write_excel(file_excel, excel_table)

    return excel_table


def insert_link_to_db(source_file):
    """
    将链接插入数据，提供后期不断自动刷新
    Args:
        source_file:

    Returns:

    """
    source_file = '%s.xlsx' % source_file
    link_list = lib_wy28.read_excel(source_file, '工作表1')
    insert_sql_list = []

    today = datetime.datetime.now()
    insert_db_time = today.strftime('%Y-%m-%d %H:%M')
    for i, link in enumerate(link_list):
        # 去表头
        if i == 0:
            continue

        valuelist = link
        valuelist.append(insert_db_time)
        valuelist_plus = lib_wy28.db_plus_valuelist(valuelist)
        insert_sql = 'insert into %s (%s) values (%s)' % \
                     (conf.MYSQL_TABLE, ','.join(DB_COL), ','.join(valuelist_plus))
        insert_sql_list.append(insert_sql)

    lib_wy28.db_execute_sqllist(insert_sql_list)


def run_doc_deadlink_todb(lan_version):
    """
    测试文档(html)死链
    Returns:

    """
    filename_sub = lan_version.replace('/', '_').replace('.', '') + '_doc'
    f_midoutput_sub = '%s_midoutput' % filename_sub
    f_midoutput_owner_sub = '%s_midoutput_owner' % filename_sub
    f_result_sub = '%s_result' % filename_sub

    get_doc_links(lan_version, f_midoutput_sub)
    save_links(lan_version, f_midoutput_sub)
    # 去掉owner，有的时候不需要owner
    # 2020-12-10 因为需要拼接link_host等，将add_doc_owner保留但owner给默认值''
    add_doc_owner('%s.xlsx' % f_midoutput_sub, f_midoutput_owner_sub, lan_version)
    insert_link_to_db(f_midoutput_owner_sub)
    # test_deadlink('%s.xlsx' % f_midoutput_sub, f_result_sub)
    # add_doc_owner('%s.xlsx' % f_result_sub, f_result_owner_sub, lan_version)


def run_dblink_update_httpcode():
    """
    防止抓取超时
    Returns:

    """
    # ***************************对link去重，加快抓取（比如中英文有同一link，减少抓两遍）*****
    # # 对所有没有抓取的link去重
    # mysql_uniq_link = "update paddle_deadlink set is_uniq_link=1 where id in " \
    #                   "(select table_tmp.id from (select id from paddle_deadlink " \
    #                   "where (status_code like '%timed out%' or status_code='') " \
    #                   "and insert_db_time like '2020-09-16%' " \
    #                   "group by link) as table_tmp);"
    # print(mysql_uniq_link)
    # lib_wy28.db_execute_sqllist(mysql_uniq_link)

    # SQL_FILTER = "select id, link, page, status_code,link_host from paddle_deadlink " \
    #              "where (status_code like '%timed out%' or status_code='') and is_uniq_link=1 "
    # *********************************************************************************

    # ***************************按照优先级抓取（对所有没有httpcode的抓，不去重）************
    SQL_FILTER = "select id, link, page, status_code,link_host from paddle_deadlink " \
                 "where (status_code like '%timed out%' or status_code='') "
    sql_search_state = [
        # 按照优先级抓取
        # 1.所有官网内部链接（不带锚点#）【去重】
        SQL_FILTER + " and link_host in ('%s')" % conf.PADDLE_HOST + "and link not like '%#%'",
        # 2.所有Github链接【去重】
        SQL_FILTER + " and link_host in ('https://github.com')",
        # 3. 所有第三方链接
        SQL_FILTER + " and link_host not in ('https://github.com','%s')" % conf.PADDLE_HOST,
        # # 4.所有官网内部链接（带锚点#）【去重】【锚点链接太多，影响效率，暂不检测锚点链接】
        # SQL_FILTER + "and link_host in ('%s')" % conf.PADDLE_HOST + " and link like '%#%'",
    ]

    # ***************************普通抓取（逻辑简单，容易卡死）***************************
    # sql_search_state = ["select id, link, page, status_code,link_host from paddle_deadlink "
    #                     "where (status_code like '%timed out%' or status_code='') and "
    #                     "link_host not in ('https://www.tensorflow.org', "
    #                     "'https://en.wikipedia.org', 'http://en.wikipedia.org', "
    #                     "'https://zh.wikipedia.org')"]

    print(sql_search_state)
    rows = lib_wy28.db_execute_sqllist(sql_search_state)
    sql_update_state_list = []
    for row in rows:
        print(row)
        id = row[0]
        url = row[1].strip()
        source = row[2]
        try:
            # 设置TCP连接超时为3s，HTTP请求超时为7s
            r = requests.get(url, timeout=(3, 7), verify=False)
            status = str(r.status_code)
            print(status)
            if status != '200':
                print('[url request fail][%s]%s[source:%s]' % (status, url, source))

            sql_update_state = "update paddle_deadlink set status_code=\'%s\' where id=\'%s\'" % (status, id)
            print(sql_update_state)
            lib_wy28.db_execute_sqllist([sql_update_state])
            sql_update_state_list.append(sql_update_state)
        except Exception as err:
            # 发生错误时回滚
            status = str(err)
            print('[url request fail][%s]%s[source:%s]' % (status, url, source))
    return sql_update_state_list


def run_md_deadlink(repo_name, version, code_path, func):
    """
    测试GITHUB(md)死链
    Args:
        lan_version:
        code_path:

    Returns:

    """
    filename_sub = repo_name + '_' + version.replace('/', '_').replace('.', '') + '_md'
    f_midoutput_sub = '%s_midoutput' % filename_sub
    f_result_sub = '%s_result' % filename_sub

    if func == "get_link":
        get_md_links(version, code_path, f_midoutput_sub, repo_name)
        save_links(version, f_midoutput_sub)
    elif func == "all":
        get_md_links(version, code_path, f_midoutput_sub, repo_name)
        save_links(version, f_midoutput_sub)
        test_deadlink('%s.xlsx' % f_midoutput_sub, f_result_sub)
    else:
        pass


def get_pages_samples(lan_version):
    """
    获取某个版本某个语言的所有样例代码
    Args:
        lan_version:
    Returns:

    """
    # 【所有的文档上、左导航链接】wy-side-scroll
    if lan_version.startswith('zh'):
        index = '%s/documentation/docs/%s/index_cn.html' % (conf.PADDLE_HOST, lan_version)
    elif lan_version.startswith('en'):
        index = '%s/documentation/docs/%s/index_en.html' % (conf.PADDLE_HOST, lan_version)
    # 文档的所有page
    doc_pages = spider_html_block_links(index, {'class': 'wy-side-scroll'})
    print('doc[%s]page num: %d' % (lan_version, len(doc_pages)))

    for page in doc_pages:
        url = page['url']

        # 【不下载锚点页的代码】跳过锚点页面，网页迭代需要升级代码
        if '#' in url:
            continue
        # # 当检测api以外的时候加上此句，检测全部的时候注释此句
        # if 'api' in page['url']:
        #     continue

        path = conf.RESULT_SAMPLE_PATH + lan_version + \
               url.split(lan_version)[-1].replace('.html', '')
        sample_path = path
        # file = url.split(lan_version)[-1].replace('.html', '').split('/')[-1]
        # 会出现路径重名的情况
        # sample_path = path.replace(file, '')
        print(url, sample_path)
        lib_wy28.spider_doc_sample(url, sample_path)


def write_icafe():
    """
    写入icafe，一个一个写，方便看哪个负责人有问题
    Args:
        excel_table:
    ['标题', '内容', '负责人', '所属计划', '父卡片编号',
    '需求来源', '问题细分类', '所属团队', '所属方向', '所属方向-细分']
    Returns:

    """
    # 日志里：page has no link
    # 数据库里：404
    # 可能有抓取重复，如果有重复用下面那个
    # mysql_bug_filter = ["select page, page_name, link, link_name, version, status_code "
    #                     "from paddle_deadlink where insert_db_time='2020-09-14 22:08:00' and link "
    #                     "in (select uniq_link_table.link from "
    #                     "(select link from paddle_deadlink where is_uniq_link=1 "
    #                     "and insert_db_time='2020-09-14 22:08:00' "
    #                     "and status_code='404') as uniq_link_table)"]
    mysql_bug_filter = ["select page, page_name, link, link_name, version, "
                        "status_code from paddle_deadlink "
                        "where is_uniq_link=1 and "
                        "insert_db_time='2020-09-14 22:08:00' and status_code='404'"]
    print(mysql_bug_filter)
    exit()
    excel_table = lib_wy28.db_execute_sqllist(mysql_bug_filter)
    # 将buglist从小到大排序
    for row in excel_table:
        icafe_dict = {'username': conf.ICAFE_USERNAME, 'password': conf.ICAFE_PASSWORD}
        item_dict = {}
        item_dict['type'] = 'Bug'
        item_dict['title'] = '【官网死链】%s版本%s页面的"%s"处报%s错误' % (row[4], row[1], row[3], row[5])
        item_dict['detail'] = '%s<br>1.死链所在页面：%s<br>2.死链：%s<br>' \
                               % (item_dict['title'], row[0], row[2])
        item_dict['parent'] = 11159
        item_dict["fields"] = {
            '所属计划': '飞桨项目集/Paddle/v2.0.0-beta',
            '需求来源': 'QA团队',
            '问题细分类': '易用性问题-文档/官网有误',
            '所属团队': '产品团队',
            '所属方向': '文档',
            '所属方向-细分': 'API文档'
        }

        # 一个一个写，所以只装一个item_dict
        icafe_dict['issues'] = [item_dict]
        # print(icafe_dict)
        result = requests.post(conf.ICAFE_API_NEWCARD, data=json.dumps(icafe_dict))
        result_dict = json.loads(result.text)
        print('写入Icafe【%s】：%s' % (result_dict['status'], result_dict['message']))
        time.sleep(1)


def write_icafe_exchange_enzh():
    """
    中英文切换问题
    """
    # # API只有英文，没有中文
    # mysql_bug_filter = ["select table2.page as page, table2.link_key as link_key from "
    #                     "(select DISTINCT(page), link_key from paddle_deadlink where version='zh/2.0-beta' "
    #                     "and insert_db_time like '2020-09-16%') "
    #                     "as table1 "
    #                     "right JOIN "
    #                     "(select DISTINCT(page), link_key from paddle_deadlink where version='en/2.0-beta' "
    #                     "and insert_db_time like '2020-09-16%') "
    #                     "as table2 "
    #                     "ON table1.link_key=table2.link_key "
    #                     "where table1.page is NULL and table2.page not like '%#%' and  table2.page like '%api%'"]
    # API只有中文，没有英文
    mysql_bug_filter = ["select table1.page as page, table1.link_key as link_key from "
                        "(select DISTINCT(page), link_key from paddle_deadlink where version='zh/2.0-beta' "
                        "and insert_db_time like '2020-09-16%') "
                        "as table1 "
                        "left JOIN "
                        "(select DISTINCT(page), link_key from paddle_deadlink where version='en/2.0-beta' "
                        "and insert_db_time like '2020-09-16%') "
                        "as table2 "
                        "ON table1.link_key=table2.link_key "
                        "where table2.page is NULL and table1.page not like '%#%' and  table1.page like '%api%'"]

    print(mysql_bug_filter)
    excel_table = lib_wy28.db_execute_sqllist(mysql_bug_filter)
    # 将buglist从小到大排序
    for row in excel_table:
        icafe_dict = {'username': conf.ICAFE_USERNAME, 'password': conf.ICAFE_PASSWORD}
        item_dict = {}
        item_dict['type'] = 'Bug'
        item_dict['title'] = '【官网中英文切换问题】develop版本%s页面存在中英文切换问题' % row[1]
        item_dict['detail'] = '%s<br>问题链接：%s' % (item_dict['title'], row[0])
        item_dict['parent'] = 11159
        item_dict["fields"] = {
            '所属计划': '飞桨项目集/Paddle/v2.0.0-beta',
            '需求来源': 'QA团队',
            '问题细分类': '易用性问题-文档/官网有误',
            '所属团队': '产品团队',
            '所属方向': '文档',
            '所属方向-细分': 'API文档'
        }

        # 一个一个写，所以只装一个item_dict
        icafe_dict['issues'] = [item_dict]
        # print(icafe_dict)
        result = requests.post(conf.ICAFE_API_NEWCARD, data=json.dumps(icafe_dict))
        result_dict = json.loads(result.text)
        print('写入Icafe【%s】：%s' % (result_dict['status'], result_dict['message']))
        time.sleep(1)


# ***************************test warning  start**********************
def spider_html_block_warning(page, block_filter):
    """
    根据html块（某个div）获取某个页面里的所有链接
    Args:
        page:
        block_filter:
    Returns:

    """
    links = []
    # 抓取
    page = page.strip()
    print(page)
    try:
        # 设置TCP连接超时为3s，HTTP请求超时为7s
        r = requests.get(page, timeout=(3, 7), verify=False)
        status = r.status_code
        if status == 200:
            print('[url request success][%d]%s' % (status, page))
        else:
            print('[url request fail][%d]%s' % (status, page))
            # return links
            # exit(1)
    except Exception as err:
        # 发生错误时回滚
        print('[url request fail][%s]%s' % (str(err), page))
        # return links
        # exit(1)

    soup = BeautifulSoup(r.text, 'html5lib')
    try:
        for a in soup.find(attrs=block_filter).find_all(attrs='system-message'):
            try:
                # basename = a['href']
                # url = filter_url(page, basename, block_filter)
                # if url != '':
                link_dict = {}
                link_dict['name'] = a.get_text()
                print(a.get_text())
                link_dict['url'] = page
                links.append(link_dict)
                # 有"去重"需求时，一个页面有一个warning就行
                break
            except Exception as err:
                print('a without href[error:%s][page:%s]' % (a, page))
    except Exception as err:
        print('[page has no link][page:%s]' % page)
    return links


def get_doc_warning(lan_version, filename_sub):
    """
    获取某个版本某个语言的所有文档链接
    Args:
        lan_version:
    Returns:

    """
    pages_links = []
    # 【所有的文档上、左导航链接】wy-side-scroll
    if lan_version.startswith('zh'):
        index = '%s/documentation/docs/%s/index_cn.html' % (conf.PADDLE_HOST, lan_version)
    elif lan_version.startswith('en'):
        index = '%s/documentation/docs/%s/index_en.html' % (conf.PADDLE_HOST, lan_version)
    # 文档的所有page
    doc_pages = spider_html_block_links(index, {'class': 'wy-side-scroll'})
    print('doc[%s]page num: %d' % (lan_version, len(doc_pages)))

    for i, page in enumerate(doc_pages):
        if 'api' not in page['url'] or '#' in page['url']:
            continue
        page_dict = {}
        page_dict['lan_version'] = lan_version
        page_dict['page'] = page
        # 文档主体内容:document
        center_links = []
        print('[%d-%d]' % (len(doc_pages), i))
        center_links = spider_html_block_warning(page['url'], {'class': 'document'})
        page_dict['links'] = center_links
        # 【暂不用】右侧锚点:navigation-toc
        # center_links = spider_html_block_links(page['url'], {'class': 'navigation-toc'})
        pages_links.append(page_dict)

    file_pkl = '%s.pkl' % filename_sub
    # 清理环境
    if os.path.isfile(file_pkl):
        os.remove(file_pkl)
    # 存dict到pkl
    f = open(file_pkl, 'wb')
    pickle.dump(pages_links, f)
    f.close()
    return pages_links


def run_doc_warning(lan_version):
    """
    测试文档(html)死链
    Returns:

    """
    filename_sub = lan_version.replace('/', '_').replace('.', '') + '_doc'
    f_midoutput_sub = '%s_midoutput' % filename_sub
    f_midoutput_owner_sub = '%s_midoutput_owner' % filename_sub

    get_doc_warning(lan_version, f_midoutput_sub)
    save_links(lan_version, f_midoutput_sub)
    add_doc_owner('%s.xlsx' % f_midoutput_sub, f_midoutput_owner_sub, lan_version)
    insert_link_to_db(f_midoutput_owner_sub)


def write_icafe_doc_warning():
    """
    写入doc warning的bug
    Args:
        excel_table:
    ['标题', '内容', '负责人', '所属计划', '父卡片编号',
    '需求来源', '问题细分类', '所属团队', '所属方向', '所属方向-细分']
    Returns:

    """
    mysql_bug_filter = ["select page, page_name, version, count(*) as num "
                        "from paddle_deadlink "
                        "where insert_db_time='2020-09-18 16:46:00' and page not like '%#%' "
                        "group by page"]
    print(mysql_bug_filter)
    excel_table = lib_wy28.db_execute_sqllist(mysql_bug_filter)
    # 将buglist从小到大排序
    for row in excel_table:
        icafe_dict = {'username': conf.ICAFE_USERNAME, 'password': conf.ICAFE_PASSWORD}
        item_dict = {}
        item_dict['type'] = 'Bug'
        item_dict['title'] = '【官网文档WARNING】%s版本%s页面存在 %d 处WARNING' % (row[2], row[1], row[3])
        item_dict['detail'] = '%s<br>1.WARNING所在页面：%s<br>' \
                               % (item_dict['title'], row[0])
        item_dict['parent'] = 11799
        item_dict["fields"] = {
            '所属计划': '飞桨项目集/Paddle/v2.0.0-beta',
            '需求来源': 'QA团队',
            '问题细分类': '易用性问题-文档/官网有误',
            '所属团队': '产品团队',
            '所属方向': '文档',
            '所属方向-细分': 'API文档',
            '优先级': 'P1-严重问题 High',
            '负责人': 'chenlong21'
        }

        # 一个一个写，所以只装一个item_dict
        icafe_dict['issues'] = [item_dict]
        # print(icafe_dict)
        result = requests.post(conf.ICAFE_API_NEWCARD, data=json.dumps(icafe_dict))
        result_dict = json.loads(result.text)
        print('写入Icafe【%s】：%s' % (result_dict['status'], result_dict['message']))
        time.sleep(1)
# ***************************test warning  end**********************

# ***************************test run_doc_content_has_keys  start**********************


def spider_html_block_has_keys(page, block_filter, key_list):
    """
    根据html块（某个div）获取某个页面里的所有链接
    Args:
        page:
        block_filter:
    Returns:

    """
    links = []
    # 抓取
    page = page.strip()
    print(page)
    try:
        # 设置TCP连接超时为3s，HTTP请求超时为7s
        r = requests.get(page, timeout=(3, 7), verify=False)
        status = r.status_code
        if status == 200:
            print('[url request success][%d]%s' % (status, page))
        else:
            print('[url request fail][%d]%s' % (status, page))
            # return links
            # exit(1)
    except Exception as err:
        # 发生错误时回滚
        print('[url request fail][%s]%s' % (str(err), page))
        # return links
        # exit(1)

    soup = BeautifulSoup(r.text, 'html5lib')

    try:
        content = soup.find(attrs=block_filter).get_text()
        for i, key in enumerate(key_list):
            # if key in content:
            if re.findall(key, content):
                link_dict = {}
                link_dict['name'] = 'has key【%s】' % key
                link_dict['url'] = page
                links.append(link_dict)
    except Exception as err:
            print('a without href[error:%s][page:%s]' % (str(err), page))

    return links


def get_doc_content_has_keys(lan_version, filename_sub, key_list):
    """
    获取某个版本某个语言的所有文档链接
    Args:
        lan_version:
    Returns:

    """
    pages_links = []
    # 【所有的文档上、左导航链接】wy-side-scroll
    if lan_version.startswith('zh'):
        index = '%s/documentation/docs/%s/index_cn.html' % (conf.PADDLE_HOST, lan_version)
    elif lan_version.startswith('en'):
        index = '%s/documentation/docs/%s/index_en.html' % (conf.PADDLE_HOST, lan_version)
    # 文档的所有page
    doc_pages = spider_html_block_links(index, {'class': 'wy-side-scroll'})
    print('doc[%s]page num: %d' % (lan_version, len(doc_pages)))

    for i, page in enumerate(doc_pages):
        # 去除锚点页
        if '#' in page['url']:
            continue
        # # 【如果检测全部可以注释以下两行】跳过非api
        # if 'api' not in page['url']:
        #     continue
        page_dict = {}
        page_dict['lan_version'] = lan_version
        page_dict['page'] = page
        # 文档主体内容:document
        center_links = []
        print('[%d-%d]' % (len(doc_pages), i))
        center_links = spider_html_block_has_keys(page['url'], {'class': 'document'}, key_list)
        page_dict['links'] = center_links
        # 【暂不用】右侧锚点:navigation-toc
        # center_links = spider_html_block_links(page['url'], {'class': 'navigation-toc'})
        pages_links.append(page_dict)

    file_pkl = '%s.pkl' % filename_sub
    # 清理环境
    if os.path.isfile(file_pkl):
        os.remove(file_pkl)
    # 存dict到pkl
    f = open(file_pkl, 'wb')
    pickle.dump(pages_links, f)
    f.close()
    return pages_links


def run_doc_content_has_keys(lan_version, key_list):
    """
    测试文档(html)内容中是否含有指定的关键字
    Returns:
        含有关键字的url
    """
    filename_sub = lan_version.replace('/', '_').replace('.', '') + '_doc'
    f_midoutput_sub = '%s_midoutput' % filename_sub

    get_doc_content_has_keys(lan_version, f_midoutput_sub, key_list)
    save_links(lan_version, f_midoutput_sub)
# ***************************test run_doc_content_has_keys  start**********************


def get_diff_link(excel_new, excel_old, excel_diff_path):
    """
    get links which in excel_new but not in excel_old
    """

    # 清理环境
    if os.path.isfile(excel_diff_path):
        os.remove(excel_diff_path)

    excel_table = []
    changed_link = []
    #get old link
    link_old_list = lib_wy28.read_excel(excel_old, '工作表1')
    url_old = []
    for i, link in enumerate(link_old_list):
        # 去表头
        if i == 0:
            continue
        # 深拷贝，去None，方便join
        for j, item in enumerate(link):
            if (j==1) and (item is not None):
                url_old.append(item)
                break
    #get new link
    link_new_list = lib_wy28.read_excel(excel_new, '工作表1')
    for i, link in enumerate(link_new_list):
        # 去表头
        if i == 0:
            excel_table.append(link)
            continue
        # 深拷贝，去None，方便join
        link_line = []
        for j, item in enumerate(link):
            if (j==1) and (item is not None) and (item not in url_old):
                for v in link:
                    if v is None:
                       v = ""
                    link_line.append(v)
                excel_table.append(link_line)
                break
    # 存excel
    lib_wy28.write_excel(excel_diff_path, excel_table)
    

def white_url(url):
    """
    check is url in white lists
    """
    for line in LIST_WHITE:
        if (len(line) > 0) and (line in url):
            print("WHITE URL:", line, url)
            return True
    return False


if __name__ == "__main__":
    # 运行命令：python test_deadlink.py run_doc_deadlink_todb
    VERSION_ZH = 'zh/2.0'
    VERSION_EN = 'en/2.0'

    args = parse_args()

    assert args.func in ['all', 'get_link', 'diff_link', 'check_link'], 'args.func must be [all|get_link|diff_link|check_link]'
    if args.func in ['get_link', 'all']:
        assert args.repo not in [None, ''], 'args.repo is "" or None'
        assert args.branch not in [None, ''], 'args.branch is "" or None'
        assert args.code_path not in [None, ''], 'args.code_path is "" or None'
        run_md_deadlink(args.repo, args.branch, args.code_path, args.func)
    elif args.func == 'diff_link':
        assert args.link_new_file not in [None, ''], 'args.link_new_file is "" or None'
        assert args.link_old_file not in [None, ''], 'args.link_old_file is "" or None'
        assert args.link_diff_file not in [None, ''], 'args.link_diff_file is "" or None'
        get_diff_link(args.link_new_file, args.link_old_file, args.link_diff_file)
    elif args.func == 'check_link':
        assert args.link_file not in [None, ''], 'args.link_file is "" or None'
        assert args.link_res_file not in [None, ''], 'args.link_res_file is "" or None'
        test_deadlink(args.link_file, args.link_res_file)
    else:
        pass 

