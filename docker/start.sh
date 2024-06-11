if [ -z "$1" ]; then
    device=0
else
    device=$1
fi

echo "start dockergpu device: $device"

cd ..
docker run -it --rm --name hrac --gpus "device=$device" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace hrac_img "bash"