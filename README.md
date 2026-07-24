# 🔌 Lightweight Electronic Component Recognition System

## MobileNetV2 + Structured Pruning + INT8 Quantization for Edge Deployment

A lightweight electronic component recognition system designed for edge
deployment scenarios.

This project implements a complete deep learning engineering workflow:

-   MobileNetV2 lightweight classification backbone
-   Structured pruning for model compression
-   INT8 dynamic quantization for inference optimization
-   TorchScript model export
-   Streamlit interactive deployment

The project focuses on transforming a research model into a practical,
efficient and deployable intelligent recognition system.

------------------------------------------------------------------------

# ✨ Project Highlights

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  Lightweight Architecture            MobileNetV2 based electronic
                                      component recognition model

  Compression Optimization            Structured pruning removes
                                      redundant channels and reduces
                                      computation

  Quantization Deployment             INT8 dynamic quantization improves
                                      inference efficiency

  Engineering Pipeline                Complete workflow from training,
                                      optimization, evaluation to
                                      deployment

  Interactive Application             Streamlit Web interface for
                                      real-time recognition
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🎯 Engineering Motivation

Deep learning models often achieve high accuracy with increasing
computational cost. However, real-world edge deployment requires models
with:

-   Lower memory usage
-   Faster inference speed
-   Hardware-friendly structure
-   Minimal accuracy loss

This project solves these deployment challenges through systematic model
compression and optimization.

------------------------------------------------------------------------

# 🧠 Solution Pipeline

## 1. Baseline Model

MobileNetV2 is selected as the backbone because of its efficient
depthwise separable convolution design and strong suitability for edge
devices.

## 2. Structured Pruning

Channel-level structured pruning is applied to remove redundant network
structures.

Advantages:

-   Reduces actual computation
-   Improves deployment efficiency
-   Maintains hardware compatibility

## 3. INT8 Quantization

Dynamic quantization is introduced to reduce model storage and
accelerate inference.

Optimization goals:

-   Smaller model footprint
-   Faster CPU inference
-   Stable recognition accuracy

## 4. Deployment System

A Streamlit application integrates the optimized model and provides:

-   Image upload
-   Component prediction
-   Confidence visualization
-   Deployment result demonstration

------------------------------------------------------------------------

# 📌 Project Demo

![Deployment Demo](results/demo/deployment_demo.png)

The system receives electronic component images and returns:

-   Recognition category
-   Confidence score
-   Optimized model inference result

Example:

-   Category: armature
-   Confidence: 99.30%

------------------------------------------------------------------------

# 📊 Engineering Results
## Model Compression Performance Evaluation

To validate the impact of model compression techniques, the inference performance is compared across three model variants:

- Original MobileNetV2 baseline
- MobileNetV2 with structured pruning
- MobileNetV2 with INT8 quantization


![Optimization Comparison](results/inference/double_sample_2row_compare.png)
The final system achieves:

-   Lightweight model deployment
-   Compression through structured pruning
-   INT8 inference optimization
-   Complete end-to-end deployment workflow

------------------------------------------------------------------------

# 🏗️ Repository Structure

``` text
.
├── configs
├── data
├── deployment
├── models
│   ├── baseline
│   ├── pruning
│   └── quantized
├── results
├── src
│   ├── train
│   ├── pruning
│   ├── quantization
│   └── evaluation
└── README.md
```

------------------------------------------------------------------------

# 🚀 Quick Start

``` bash
pip install -r requirements.txt
```

Run Web Demo:

``` bash
streamlit run deployment/app.py
```

------------------------------------------------------------------------

# 🛠️ Technology Stack

-   Python
-   PyTorch
-   TorchVision
-   MobileNetV2
-   Structured Pruning
-   INT8 Quantization
-   TorchScript
-   Streamlit

------------------------------------------------------------------------

# 💡 Engineering Thinking

The core value of this project is not only training a classifier, but
completing the full engineering process:

Model Design → Optimization → Compression → Deployment → Application

It demonstrates the ability to convert deep learning algorithms into
efficient and practical systems.
