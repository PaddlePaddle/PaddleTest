docker rm -f 2.4_jx
nvidia-docker run  --name 2.4_jx --shm-size=128G  --network=host --cap-add=SYS_ADMIN -v $PWD:/paddle -v /home:/home -v /ssd1:/ssd1 -v /ssd2:/ssd2 -w /paddle -it registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.2-trt8.0-gcc82 /bin/bash
