cd ..
docker run -it --rm --name hrac --gpus "device=0" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace hrac_img "bash"