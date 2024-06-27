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

echo "start dockergpu device: $device"
echo "start docker name: hrac_$docker_container_idx"

cd ..
docker run -it --rm --name hrac_$docker_container_idx --gpus "device=$device" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace hrac_safety_img "bash"