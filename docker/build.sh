if [ -z "$1" ]; then
    additional_name=
else
    additional_name=$1
fi

cd ..
# check if safety-gym exist
if [ -z "$(ls -A safety-gym)" ]; then
   echo "safety-gym folder doesnt exist!!!"
else
   docker build -t mb_rce_img$additional_name -f docker/dockerfile .
fi