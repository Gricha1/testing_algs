if [ -z "$1" ]; then
    device=0
else
    device=$1
fi

if [ -z "$2" ]; then
    docker_container_idx=0
else
    docker_container_idx=$2
fi

if [ -z "$3" ]; then
    additional_name=
else
    additional_name=$2
fi

cd ..
docker run -it --rm --name mb_rce_img_$docker_container_idx --gpus "device=$device" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace mbppol_img$additional_name "bash"