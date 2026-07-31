
if [ -z "$1" ]; then
    gpus=0
else
    gpus=$1
fi


if [ -z "$2" ]; then
    container_postfix=
else
    container_postfix=$2
fi


image_name=safe_dynalang_img
container_name=safe_dynalang_$container_postfix

echo image name --- $image_name
cd ..

# check safety gymnasium
if [ ! -d "safety-gymnasium" ]; then
    echo "safety-gymnasium is not founded, downloading ..."
    git clone https://github.com/PKU-Alignment/safety-gymnasium.git
else
    echo "safety-gymnasium exists"
fi


# Craftax
if [ ! -d "SafeDreamer/embodied/envs/Craftax" ]; then
    echo "Craftax is not founded, downloading ..."
    git clone -b craftax_python_3.8 https://github.com/Gricha1/testing_envs SafeDreamer/embodied/envs/Craftax
else
    echo "Craftax exists"
fi

# CrafText
if [ ! -d "SafeDreamer/embodied/envs/CrafText" ]; then
    echo "CrafText is not founded, downloading ..."
    git clone -b cmdp https://github.com/ZoyaV/CrafText SafeDreamer/embodied/envs/CrafText
else
    echo "CrafText exists"
fi


if [ -d "logdir" ]; then
    echo container name --- $container_name
else
    mkdir logdir
    echo create log dir 
    echo container name --- $container_name
fi
echo gpus in docker --- $gpus

docker run -it --rm --name $container_name --memory="200g" --gpus "device=$gpus" --env WANDB_API_KEY=$WANDB_API_KEY -v $(pwd):/usr/home/workspace -v $(pwd)/logdir:/root/logdir -v $(pwd)/docker/entrypoint.sh:/usr/local/bin/safe_dynalang_entrypoint.sh:ro --entrypoint /usr/local/bin/safe_dynalang_entrypoint.sh $image_name
