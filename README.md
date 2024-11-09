# Install

- conda create --name arrg_proprocessing python=3.9 -y
- conda activate arrg_proprocessing
- conda install -c conda-forge spacy
  - python -m spacy download en_core_web_sm
- pip install datasets
- pip install tqdm