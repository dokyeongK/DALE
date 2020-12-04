# DALE : Dark Region-Aware Low-light Image Enhancement (BMVC 2020)
## Introduction
This repository is an official Pytorch implement of the paper "DALE: Dark Region-Aware Low-light Image Enhancement" from [BMVC2020](https://bmvc2020.github.io/). 

📣[paper](https://arxiv.org/abs/2008.12493)

### Main Contribution

Proposed a new attention module to recognize dark areas.

Propose a novel low-light enhancement method using the proposed dark-aware [visual attention](#visual-attention-map).


### Network
![DALE_network](https://user-images.githubusercontent.com/28749482/90978985-a7b41300-e58c-11ea-9b81-c0b6e4afdcf4.JPG)

### Visual Attention Map
![DALE_visual_attention_map](https://user-images.githubusercontent.com/28749482/90980969-50686f80-e599-11ea-8ba4-526152f79397.JPG)

## Code
model & test code : Now Updated! 🎈

### Requirements
![](https://img.shields.io/badge/OS-Win10-green.svg) 
```
- Python 3.6
- Pytorch >= 1.0.0
- Visdom
- PIL
- SciPy
```

### Demo


## Results

### Visual Results
Qualitative Result. `DALE` & `DALEGAN` -> ours🧐
![DALE_visual](https://user-images.githubusercontent.com/28749482/90978820-8acb1000-e58b-11ea-87de-e93a430aff39.JPG)

LOE Result Visualization (Our Result : Less distortion & Sufficient Light 😍)
![DALE_LOE](https://user-images.githubusercontent.com/28749482/90978871-ed241080-e58b-11ea-929c-b30087ed5058.JPG)

### Quantitative Results
![DALE_table](https://user-images.githubusercontent.com/28749482/90978461-155e4000-e589-11ea-9cab-52024af8daf6.JPG)
