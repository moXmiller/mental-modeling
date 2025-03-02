This is the repository for our paper "Beyond Pattern Recognition: Probing Mental Representations of LMs". All data can be found under [moritzmiller/mental-modelling](https://huggingface.co/datasets/moritzmiller/mental-modeling).
The directory consists of three groups of scripts:

1. scripts for executing the experiments include `mm_mental_models.py`, `mm_text_additional_models.py`, `mm_text_direct_additional.py`, `mm_text_mental.py`, `mm_vlm.py`, `mm_vlm_mental.py`, `mm_vlm_sbs.py`
2. scripts for preprocessing and dataset creation are `mm_extract_correct_json.py`, `mm_store_graphs.py`, `mm_structured_mwps.py`, `mm_text_to_lf.py`, `mm_utils.py`, of which several refer to the template stored under `data`
3. scripts for creating the \textsc{MathWorld} worldmodel and analysing results stored in `mathworld_repo` and `mm_statistics.py`
   
All our experiments can be run on 4 NVIDIA A100 80GB GPUs. To reproduce our results, use the following setup:
```
conda install -y pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install vllm==0.6.3.post1
conda install -y -c conda-forge graphviz python-graphviz
```
