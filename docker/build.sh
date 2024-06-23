cd ..
 check if safety-gym exist
if [ -z "$(ls -A safety-gym)" ]; then
   echo "safety-gym folder doesnt exist!!!"
else
   docker build -t hrac_safety_img -f docker/dockerfile .
fi