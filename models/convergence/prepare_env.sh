export no_proxy=bcebos.com

# python
ldconfig;
echo python_version;
echo export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:/usr/local/ssl/lib:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 >> /etc/profile
echo export PATH=/opt/_internal/cpython-3.8.0/bin/:/usr/local/ssl:/usr/local/go/bin:/root/gopath/bin:/usr/local/gcc-8.2/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin >> /etc/profile
source /etc/profile
python -c 'import sys; print(sys.version_info[:])';

# paddle
python -m pip install -U https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda112-Trton-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
python -c 'import paddle;print(paddle.__version__,paddle.version.commit)'

# nfs
# ubuntu
if [ -f "/etc/lsb-release" ];then
apt-get update -y
apt-get install nfs-common -y
# centos
elif [ -f "/etc/redhat-release" ];then
yum update -y
yum install nfs-utils -y
fi
cd /paddle
if [ ! -d "data" ];then
mkdir data
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 10.:/ data
fi
