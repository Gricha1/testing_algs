cd ..
docker run -it --rm --name mb_rce_img --gpus "device=0" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace mbppol_img "bash"