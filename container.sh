docker run -itd -p 8888:8888 --name nb -v .:/notebook --device nvidia.com/gpu=all python
docker container exec nb pip install jupyter numpy matplotlib accelerate transformers==5.4.0 datasets tokenizers trl
docker container exec nb pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
docker container exec nb jupyter notebook --ip 0.0.0.0 --allow-root /notebook
