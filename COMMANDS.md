# Build Docker
docker build --network=host -t his .


# RTX 4090D x 2
sudo docker save his | gzip > /home/ubuntu/Desktop/his.tar.gz

cd /home/ubuntu/Desktop/HIS

docker run --gpus all -it --rm --network=host \
    -v $PWD:/workspace \
    -v /home/ubuntu/Desktop/Stable-Diffusion-3-Medium:/models/sd3 \
    -v /home/ubuntu/Desktop/FLUX.1-dev:/models/flux \
    -e SD3_MODEL_PATH=/models/sd3 \
    -e FLUX_MODEL_PATH=/models/flux \
    -e HF_HUB_OFFLINE=1 \
    his bash

# RTX 4090D x 2 Parllel Execution
docker run --gpus '"device=0"' -it --rm --network=host \
    -v $PWD:/workspace \
    -v /home/ubuntu/Desktop/Stable-Diffusion-3-Medium:/models/sd3 \
    -v /home/ubuntu/Desktop/FLUX.1-dev:/models/flux \
    -e SD3_MODEL_PATH=/models/sd3 \
    -e FLUX_MODEL_PATH=/models/flux \
    -e HF_HUB_OFFLINE=1 \
    his bash


docker run --gpus '"device=1"' -it --rm --network=host \
    -v $PWD:/workspace \
    -v /home/ubuntu/Desktop/Stable-Diffusion-3-Medium:/models/sd3 \
    -v /home/ubuntu/Desktop/FLUX.1-dev:/models/flux \
    -e SD3_MODEL_PATH=/models/sd3 \
    -e FLUX_MODEL_PATH=/models/flux \
    -e HF_HUB_OFFLINE=1 \
    his bash

# A100 x 2
gunzip -c /path/to/his.tar.gz | docker load
docker images | grep his

cd /path/to/HIS

docker run --gpus '"device=0"' -it --rm --network=host \
    -v $PWD:/workspace \
    -v /home/ubuntu/Desktop/Stable-Diffusion-3-Medium:/models/sd3 \
    -v /home/ubuntu/Desktop/FLUX.1-dev:/models/flux \
    -e SD3_MODEL_PATH=/models/sd3 \
    -e FLUX_MODEL_PATH=/models/flux \
    -e HF_HUB_OFFLINE=1 \
    his bash


docker run --gpus '"device=1"' -it --rm --network=host \
    -v $PWD:/workspace \
    -v /home/ubuntu/Desktop/Stable-Diffusion-3-Medium:/models/sd3 \
    -v /home/ubuntu/Desktop/FLUX.1-dev:/models/flux \
    -e SD3_MODEL_PATH=/models/sd3 \
    -e FLUX_MODEL_PATH=/models/flux \
    -e HF_HUB_OFFLINE=1 \
    his bash