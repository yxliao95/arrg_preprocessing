# Document

## Install

nvidia/Llama-3.1-Nemotron-70B-Instruct-HF is tested on Transformers v4.44.0, torch v2.4.0
We are using CUDA/12.4

- `conda create -n arrg_sentgen python=3.11 -y`
- `conda activate arrg_sentgen`
- `conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y`
- `pip install transformers` (==4.46.0)
- `pip install accelerate` (==1.0.1)

Download LLM:

- `pip install --upgrade huggingface_hub`
- `huggingface-cli login --token your_access_token_here`
    - Get the access token from: https://huggingface.co/settings/tokens
- `huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /your/specific/path`