docker run -itd -p 8888:8888 --name nb -v .:/notebook --device nvidia.com/gpu=all python
docker container exec nb pip install jupyter numpy matplotlib transformers datasets tokenizers trl
docker container exec nb pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
docker container exec nb jupyter notebook --ip 0.0.0.0 --allow-root /notebook
