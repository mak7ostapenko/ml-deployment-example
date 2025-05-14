# ml-deployment-example
ML deployment with FastAPI and ONNX

The aim of this repository is to provide starter code for your ML models deployment or to serve as a demo base for your research project.

I am going to work with depth estimation and as a model [UniDepthV2](https://github.com/lpiccinelli-eth/unidepth) is seletected. 

## 🌲 Repository Structure

```
/src
```
Generated with `tree .`.

## 🛠️ Setup

*This was tested on Ubuntu 24.04 LTS with Python 3.12.3 and 16GB RAM*

### 🐍 Python Environment

```bash
python3 -m venv .env/
source .env/bin/activate
pip install -r requirements.txt
```

### 💾 Data


## 🚀 Usage

### Serve with Docker

### Serve with tmux
This serve is not made for production use, but good for testing, especcially on some remote device 

```bash
tmux new -s SESSION_NAME

# inside of it run service
./run_service.sh

# to exit session window use
ctrl+b + d

# to open session again
tmux a -t SESSION_NAME
```

More usefull commands to learn [tmux](https://gist.github.com/MohamedAlaa/2961058)

### Add your own model

Here is how I added the model

First, copy model's repository into `src/` folder.

```bash
cd src/
# clone your target repository
git clone https://github.com/lpiccinelli-eth/UniDepth.git

# then add it to .gitignore
cd ../
echo src/UniDepth >> .gitignore
```
**Note**, also you can add repository using git module.

Second, follow model's specefic instructions to launch it.

### Transform the model into ONNX format
In this case, I am going to use already provided code to export into `.onnx` format.

For completness going to add the UniDepth specific instruction.
```bash
# to run Unideptth in the docker container do
cp Dockerfile_unidepth src/UniDepth/Dockerfile

# build container and export model to .onnx format
cd src/UniDepth

docker build -t dev_unidepth .
docker run -it --gpus all \
    --name cont_ud_1 \
    -v "$(pwd):/home/depth_estimation" \
    dev_unidepth

# inside of the container
python3 ./unidepth/models/unidepthv2/export.py --version v2 --backbone vits --shape 462 630 --output-path unidepthv2_vits_462_630.onnx
```

Finally, move the model to checkpoints folder.
```bash
mkdir checkpoints/
mv src/UniDepth/unidepthv2_vits_462_630.onnx checkpoints/
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**NOTE**. [UniDepthV2](https://github.com/lpiccinelli-eth/unidepth) is released under [Creatives Common BY-NC 4.0 license](https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE).