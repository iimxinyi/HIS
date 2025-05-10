# Hybrid Inference Scheme

**Target:** The aim is to explore the misalignment between users' intentions and the generated contents (i.e., the combined effects of common inference step and the similarity between public and personal prompts), with a focus on guiding the design of an efficient hybrid inference scheme.

**Paper:** "Diffusion-Based Hybrid Inference in MEC-Enabled AI-Generated Content Networks" --submitted to IEEE Journal on Selected Areas in Communications (IEEE JSAC)

**Experimental Platform:** Our experiments are performed on an Ubuntu 20.04 system equipped with an Intel Xeon Gold 6248R CPU and an NVIDIA A100 GPU.

![image](/Files/HIS-Overview.png)


## 1 Environment Setup

Create a new conda environment.

```shell
conda create --name LVM python==3.10
```


## 2 Activate Environment

Activate the created environment.

```shell
source activate LVM
```


## 3 Install Required Packages

ubuntu==20.04  cuda==11.8
```shell
pip install torch==2.4.1
pip install sentence-transformers==3.1.1
pip install diffusers==0.30.3
pip install transformers==4.44.2
pip install accelerate==0.34.2
pip install protobuf==5.28.2
pip install sentencepiece==0.2.0
pip install openai-clip==1.0.1
pip install torchvision==0.19.1
pip install openpyxl==3.1.5
```
Then you should get an env like:
```shell
Package                  Version
------------------------ --------------------
accelerate               0.34.2
calflops                 0.3.2
certifi                  2025.1.31
charset-normalizer       3.4.1
diffusers                0.30.3
et_xmlfile               2.0.0
filelock                 3.17.0
fsspec                   2025.2.0
ftfy                     6.3.1
fvcore                   0.1.5.post20221221
huggingface-hub          0.28.1
idna                     3.10
importlib_metadata       8.6.1
iopath                   0.1.10
Jinja2                   3.1.5
joblib                   1.4.2
MarkupSafe               3.0.2
mpmath                   1.3.0
networkx                 3.4.2
numpy                    2.2.3
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.8.61
nvidia-nvtx-cu12         12.1.105
openai-clip              1.0.1
openpyxl                 3.1.5
packaging                24.2
pillow                   11.1.0
pip                      25.0
portalocker              3.1.1
protobuf                 5.28.2
psutil                   7.0.0
PyYAML                   6.0.2
regex                    2024.11.6
requests                 2.32.3
safetensors              0.5.2
scikit-learn             1.6.1
scipy                    1.15.1
sentence-transformers    3.1.1
sentencepiece            0.2.0
setuptools               75.8.0
sympy                    1.13.3
tabulate                 0.9.0
termcolor                3.0.1
thop                     0.1.1.post2209072238
threadpoolctl            3.5.0
tokenizers               0.19.1
torch                    2.4.1
torchsummary             1.5.1
torchvision              0.19.1
tqdm                     4.67.1
transformers             4.44.2
triton                   3.0.0
typing_extensions        4.12.2
urllib3                  2.3.0
wcwidth                  0.2.13
wheel                    0.45.1
yacs                     0.1.8
zipp                     3.21.0
```

## 4 Locate and Modify StableDiffusion3Pipeline
Open `Demo.py` in your code editor.

Hold down the `ctrl` key if you are on Linux or Windows, or the `command` key if you are on MacOS, and click on StableDiffusion3Pipeline.

![image](/Files/modify.png)

This will navigate to the file `pipeline_stable_diffusion_3.py`.

Replace `pipeline_stable_diffusion.py` with the file of the same name from this repository.


## 5 Explanation of Our Code Files

To avoid confusion, we rewrite the Negative Prompt Generator (NPG) in our code files as the NPI in our paper, which means the same thing.

`pipeline_stable_diffusion.py`: 

In line 664, the parameter "prompt" is the positive personal prompt (i.e., the user-provided prompt).

In line 669, the parameter "num_inference_steps" is the total number of inference steps used to generate a satisfied image.

In line 671, the parameter "guidance_scale" is the guidance scale in our proposed Semantic Intensity Modulator (SIM).

In line 672, the parameter "negative_prompt" is the negative personal prompt in our proposed Negative Prompt Injecor (NPI).

In line 689, the parameter "common_step" is the number of common inference steps (i.e., the shared steps).

In line 690, the parameter "prompt_unchanged" is True if there is no common inference phase, and vice versa.

In line 691, the parameter "get_intermediate_result" is True if you want to get the intermediate image in each inference step.

`1-Get-Figures`:

Used to generate images for evaluating the effectiveness of individual design elements.

`2-SIM-NPG-CLIP`:

Used to calculate the alignment score in the SIM and NPI.

`3-SIM-NPG-BRISQUE`:

Used to calculate the fidelity score in the SIM and NPI.

`4-Similarity-Sentence`:

Used to calculate the similarity score between the public and personal prompts.

`5-Similarity-CLIP`:

Used to calculate the image quality in our proposed Hybrid Inference Quality Metric (HIQM).

`6-Similarity-Fitting`:

Used to fit the HIQM function.


## 6 Explanation of Our Results

Our generated image is available in:

OneDrive: 

Baidu Netdisk: 

Image naming rules for the SIM：

```shell
scalex_prompty_seedz_wNPG/woNPG.png
x: guidance scale index
y: personal prompt index
z: seed index
wNPG: the NPI is enabled
woNPG: the NPI is not enabled
```

Image naming rules for the NPI：

```shell
promptx_seedy_wNPG/woNPG.png
x: personal prompt index
y: seed index
wNPG: the NPI is enabled
woNPG: the NPI is not enabled
```

Image naming rules for the HIQM (i.e., similarity)：

```shell
Publicx_Personaly_CommonStepz.png
x: public prompt index
y: personal prompt index
z: common inference step
```


## 7 Case Study
![image](/Files/CaseStudy.png)
















