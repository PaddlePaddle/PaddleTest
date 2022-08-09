cmds='
{"model":"alexnet"};
{"model":"googlenet"}
'

ix=0
IFS=";"
for cmd in $cmds;
do
        ix=$[$ix +1]
        #echo $cmd
        echo -e '\033[0;32m'START'\033[0m'-case-$ix
        curl -X GET -H 'Content-type: application/json; Accept: application/json; charset=UTF-8' -d "$cmd" http://10.255.103.24:8206/tool-6
        echo
        if [ $? -ne 0 ]; then
                echo -e '\033[0;31m'END'\033[0m'
        else
                echo -e '\033[0;32m'END'\033[0m'
        fi
done
