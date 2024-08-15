# $1 - gpu
# $2 - adding to container name
# $3 - image name

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
    image_name=safe_slac_img
else
    image_name=$3
fi

docker_container_name=safe_slac_$docker_container_idx
echo "start dockergpu device: $device"
echo "start docker name: $docker_container_name"
echo "start docker image: $image_name"

cd ..
docker run -d -it --rm --name $docker_container_name --gpus "device=$device" -v $(pwd):/usr/home/workspace $image_name "bash" 