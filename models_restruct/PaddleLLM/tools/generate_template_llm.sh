paddle_commit=`python -c "import paddle; print(paddle.version.commit)"`
paddlenlp_commit=`cd ./PaddleLLM && git rev-parse HEAD`
ipipe_url=https://console.cloud.baidu-int.com/devops/ipipe/workspaces/${AGILE_WORKSPACE_ID}/pipeline-builds/${AGILE_PIPELINE_BUILD_ID}/stage-builds/${AGILE_STAGE_BUILD_ID}/view
allure_url=https://ipipe-report.baidu-int.com/bos/$(basename "$PWD")/report/#behaviors

cat <<EOF > ./utils/default_template_llm.html
<meta http-equiv="Content-Type" content="text/html;charset=utf-8">
<html align='left'>
  <body>
    <div style="text-align: left;margin-left:1%;margin-right:2%;margin-top:2%;margin-bottom:2%;">
      <h3>任务链接</h3>
      <p>
        <a href="${ipipe_url}" target="_blank">
          查看ipipe任务详情
        </a>
      </p>
      <h3>环境信息</h3>
      <p> paddle commit: ${paddle_commit}</p>
      <p> paddlenlp commit: ${paddlenlp_commit}</p>
      <h3>报告说明</h3>
      <p>
        <a href="${allure_url}" target="_blank">
          查看allure结果报告
        </a>
      </p>
    </div>
  </body>
</html>
EOF