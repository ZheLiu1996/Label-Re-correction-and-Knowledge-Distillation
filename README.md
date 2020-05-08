This project is the source code for the paper "Improving Biomedical Named Entity Recognition with Label Re-correction and Knowledge Distillation", which focus on the Chemical-induced Diseases (CID) Relation Extraction subtask in BioCreative V Track 3 CDR Task.

URL for BioCreative V Track 3 CDR Task: http://biocreative.org/tasks/biocreative-v/track3-cdr/

The original data and official evaluation toolkit could be found here.

=============================environmental requirements====================================
python >=3.6
pytorch >= 1.1.0
pytorch-crf >= 0.7.2
tqdm >= 4.36.1
numpy >= 1.17.2

=============================Introduction of the code=======================================
preprocessd_data.py:convert the original data with the form of pubtator into the commom input form (e.g. Tricuspid	B-Disease)
processd_data.py:
