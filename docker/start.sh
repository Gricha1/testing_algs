cd ..
docker run -it --rm --name mbppol_cont --gpus "device=0" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace mbppol_img "bash"