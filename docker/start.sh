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
    image_name=omnisafe_img
else
    image_name=$3
fi

echo "start dockergpu device: $device"
echo "start docker name: omnisafe_$docker_container_idx"
echo "start docker image: $image_name"

cd ..
docker run -it --rm \
  --name ggorbov.omnisafe_$docker_container_idx \
  --gpus "device=$device" \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}" \
  -v "$(pwd)":/usr/home/workspace \
  -w /usr/home/workspace \
  "$image_name" \
  bash -lc 'python -c "import comet_ml" 2>/dev/null || pip install -q "comet_ml>=3.40.0"; cd /usr/home/workspace; exec bash'
