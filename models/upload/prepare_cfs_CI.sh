# $1：挂载点ip $2:cfs目录(以套件名称为存储路径) $3:本地挂载目录
unset http_proxy && unset https_proxy 
apt-get update
apt-get install nfs-common -y
echo ---cfs_folder:$2
echo ---local_folder:$3
if [ -d $3 ];then
   rm -rf $3
fi

mkdir $3
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $1:/$2 $3
export date_path=${PWD}/$3
echo "---date_path:${date_path}---"