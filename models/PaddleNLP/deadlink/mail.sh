root_path=$1
cd $root_path
#检查程序是否执行成功
if [ ! -f "%REPO%_%BRANCH%_md_diff_res.txt" ]
then
    echo "run fail"
    exit 1
fi

#输出文件，并判断是否有死链（404）
mkdir result
cd result
cp -r $root_path/%REPO%_%BRANCH%_md_diff_res* ./
FAILLINK=`grep 404 %REPO%_%BRANCH%_md_diff_res.txt | wc -l`
if [ "${FAILLINK}" -gt "0" ]
then
    echo "FAIL: " ${FAILLINK} " links"
    exit 2
else
    echo "SUCCESS"
    exit 0
fi
