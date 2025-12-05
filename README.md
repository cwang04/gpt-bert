<h1>DSC Capstone Quarter 1 Checkpoint</h1>
<p>
This is a fork of the repository for an entry to the BabyLM challenge that combined GPT and BERT language models.

The instructions in the README.md and original code had some issues so this fork has edits that correct these changes as well as added the 10M token data set for training. This also contains the .yaml files I used to create a pvc as well as pulling the repo, as well as the code to run the job to train the GPT-BERT model with hyperparameter sweep. The code for this model was not written by me, but is the codebase that I have been working with and has the necessary changes I made to get the code to work for me. The .yaml files are in the yaml folder.
</p>

<h2 align="center"><b><h3> Steps to follow to replicate hyperparameter sweeps</h3></b></h2><br>

**Need to have kubectl installed, as well as access to the nautilus cluster in order to properly run these .yaml files**

1. Make your own pvc, either run your own script or run the script in /yaml called `gpt-bert-pvc.yaml`, navigate to `/yaml` and then run the following command: `kubectl apply -f gpt-bert-pvc.yaml`
2. Wait for the pvc, to initialize and then run the following command, make sure to alter the file if you changed the pvc name, `kubectl apply -f gpt-bert-sweep-job.yaml`, even though `requirements.txt` exists, the .yaml will already install the packages, so that you don't need to run `pip install requirements.txt`.
    1. In order for this to work properly, you need access to the wandb sweep job. If you don't have access to it, you need to create your own wandb sweep job.
    2. Once you create the wandb hyper parameter sweep, make sure to update the sweep so that `sweep.yaml` is the .yaml that is attached to the sweep job.
    3. Make sure that you edit `gpt-bert-sweep-job.yaml` line 39 so that the call for the wandb agent sweep matches with wandb suggests for you to use.
    4. **(Optional)** if you feel it is necessary, you can alter the parameters in `sweep.yaml` to change the hyperparameter sweep conditions.
    5. When substeps 2.1-3 are completed, `kubectl apply -f gpt-bert-sweep-job.yaml` should run properly, continue to step 3
3. After waiting for the sweep job to launch, check your wandb. If done properly, you should start to see sweep jobs being run and the logs of the validation perplexity as well as which hyperparameters are being used.
4. Let this sweep job run in the background for as long as desired (you do have to consider the memory allocated to the pvc, if run for to long too much will be stored in `/wandb` and will run out of disk space). You can then use wandb to find which hyperparametersr returned the lowest perplexity and move on with training a model under these optimal parameters.




## The following is the original instructions from the repo, not realted to Q1 code, all of the steps have been integrated into yaml files already 
<br>





<h2 align="center"><b><h3>GPT or BERT: why not both?</h3></b></h2><br>


<p align="center">
  <b>Lucas Georges Gabriel Charpentier and David Samuel</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2410.24159"><b>Paper</b></a><br>
  <a href="https://huggingface.co/ltg/gpt-bert-babylm-base"><b>HuggingFace 100M model</b></a><br>
  <a href="https://huggingface.co/ltg/gpt-bert-babylm-small"><b>HuggingFace 10M model</b></a><br>
  <a href="https://huggingface.co/datasets/ltg/babylm-2024-baby-cosmo-fine-100m"><b>100M Dataset</b></a><br>
  <a href="https://huggingface.co/datasets/ltg/babylm-2024-baby-cosmo-fine-10m"><b>10M Dataset</b></a>
</p>

_______

<br>

<h3 align="center"><b>Abstract</b></h3><br>

We present a simple way to merge masked language modeling with causal language modeling. This hybrid training objective results in a model that combines the strengths of both modeling paradigms within a single transformer stack: GPT-BERT can be transparently used like any standard causal or masked language model. We test the pretraining process that enables this flexible behavior on the BabyLM Challenge 2024. The results show that the hybrid pretraining outperforms masked-only or causal-only models. We openly release the models, training corpora and code. 

_______

<br>

This is the official repository for our BabyLM 2024 submission: GPT-BERT.

_______

<br>

## Warning: This repository is not yet completed
Completed files/folders:
- data
- model_checkpoints
- tokenizers
- configs
- tokenizer_creation
- pretraining
- configs
- corpus_tokenization

Incomplete files/folders:

- evaluation

## Content of this repository

- `./tokenizer_creation/`: Contains scripts for creating a tokenizer.
- `./corpus_tokenization/`: Contains scripts to tokenize a corpus.
- `./pretraining/`: Contains scripts to train a pre-train a model, as well as the model file itself, utils, optimizers, and the PyTorch datasets.
- `./evaluation/`: Contains folders for each benchmark evaluated in the paper. Each folder contains scripts to do fine-tuning (when relevant) and inference as well as a data folder containing the data of the benchmark.
- `./data/`: Folder containing the raw, preprocessed, and tokenized data for pretraining.
- `./tokenizers/`: Folder containing the tokenizers created, or needed for pretraining.
- `./configs/`: Folder containing the configuration files for models.
- `./model_checkpoints/`: Folder containing the pre-trained model checkpoints.

_______

<br>

## Code to pre-train (and evaluate) a model

This is will be a general guide to pretraining the model, to find out what files to run and what they do, each subfolder will contain a README detailing its content.

1. (optional) If you do not have a tokenizer, or want to create a custom one, run the script(s) found in `tokenizer_creation`. The created tokenizers will be saved in `tokenizers` (unless otherwise specified). 
2. To tokenize the corpus, run the script in `corpus_tokenization`. The tokenized data will be saved in the folder `data` (unless otherwise specified). We tokenize before training for efficiency, but in the case this is not wanted, code will need to be adapted in the scripts found in `pretraining` (specifically the `dataset.py` file).
3. Create a config file for your model in the same style as the ones found in the `configs` folder. Otherwise, choose one of the pre-created ones.
4. To pre-train your model, run one of the `train_*.py` scripts found in the `pretraining` folder. (More details found in the folder itself)
5. (optional) If you want to evaluate your model based on the evaluations used in the paper, the different tasks and code to run the evaluation can be found in `evaluation`. Note: to be able to use each part independently of another, the model file is also included in each benchmark folder.
_______

<br>

## Please cite the following publication
```bibtex
@inproceedings{charpentier-samuel-2024-bert,
    title = "{BERT} or {GPT}: why not both?",
    author = "Charpentier, Lucas Georges Gabriel  and
      Samuel, David",
    editor = "Hu, Michael Y.  and
      Mueller, Aaron  and
      Ross, Candace  and
      Williams, Adina  and
      Linzen, Tal  and
      Zhuang, Chengxu  and
      Choshen, Leshem  and
      Cotterell, Ryan  and
      Warstadt, Alex  and
      Wilcox, Ethan Gotlieb",
    booktitle = "The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-babylm.24/",
    pages = "262--283",
}
```
