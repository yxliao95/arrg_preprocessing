## Install

- `conda create --name arrg_preprocessing python=3.9 -y`
- `conda activate arrg_preprocessing`
- `conda install -c conda-forge spacy`
  - `python -m spacy download en_core_web_sm`
- `pip install datasets`
- `pip install matplotlib`

## Process pipeline

1. Get the interpret-cxr dataset following: https://stanford-aimi.github.io/RRG24/. We use the training, validation, and test-public sets
2. Follow the notebook `0_prepare_datasets.ipynb` to process the mimic-cxr dataset
3. The preprocessing contains the following steps:
  1. Use spacy to identify sentences from reports: `0_prepare_datasets.ipynb`
  2. Use Llama 3.1 to split sentence into multipule sentences without conjunction words: `arrg_sentgen`
  3. Use spacy to tokenized the split sentences: `2_process_sentence.ipynb`
  4. Use Bioportal API to annotate the split sentences using the RadLex ontology: `3_bioportal_annotate.py`
  5. Use CXRGraph to annotate the split sentences in terms of Named Entity Recognition and Relation Extraction: `arrg_cxrgraph`
  6. Use RadCoref to annotate the split sentences in terms of Coreference Resolution: `fast-coref`
  7. Combine the results and create an aggregrated dataset
  8. Analyse the results