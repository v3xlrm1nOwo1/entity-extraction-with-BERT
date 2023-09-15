



## Entity extraction with BERT


### Model 

BERT base model (uncased)
Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in <a href='https://arxiv.org/abs/1810.04805'>this paper</a> and first released in <a href='https://github.com/google-research/bert'>this repository</a>. This model is uncased: it does not make a difference between english and English.

Disclaimer: The team releasing BERT did not write a model card for this model so this model card has been written by the Hugging Face team.


### Dataset

The Dataset is <a href='https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus'>Annotated Corpus for Named Entity Recognition
</a> From  <a href='https://www.kaggle.com/abhinavwalia95'>ABHINAV</a>
Dataset discription: 
Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set.



### Finetuning

In this project I used PyTorch and the model and dataset in the top and in the deep in traning loop I used 3 and 3e-6 for learning rate, you can fined more about  finetuning step in the model file (config.py, bert_model.py and engine.py etc..)

### Note

I did not have the resources, such as the Internet, electricity, device, etc., to train the model well and choose the appropriate learning rate, so there were no results.


> To contribute to the project, please contribute directly. I am happy to do so, and if you have any comments, advice, job opportunities, or want me to contribute to a project, please contact me <a href='mailto:V3xlrm1nOwo1@gmail.com' target='blank'>V3xlrm1nOwo1@gmail.com</a>

