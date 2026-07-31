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