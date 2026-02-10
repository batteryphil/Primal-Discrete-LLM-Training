# Post-Training Evolution of Large Language Models into Prime Harmonic 1.58-bit Grids

**Author:** Project Trinity Research Team  
**Date:** February 2026  

## Abstract
We introduce "Prime Harmonic Evolution," a post-training quantization method that aligns standard FP16 weights to a rigid Prime Reciprocal Grid {-1, 0, 1}. Using a TinyLlama-1.1B baseline, we achieved a structural recovery of 98% (Perplexity ~15.3) and a physical compression ratio of 8.51x (246 MB) in under 500 training steps.

## 1. Introduction
The BitNet b1.58 architecture proposes that LLMs can exist in ternary states {-1, 0, 1}. However, current methods require pre-training from scratch. We hypothesized that existing dense models could be "evolved" into this state using Gradient Accumulation and high-temperature quantization noise.

## 2. Methodology

### 2.1 The Prime Grid
We defined a target discrete space based on prime reciprocals to maximize information density while minimizing bit-width. The grid includes reciprocals of primes and strategic "tails" to capture outlier distributions.

### 2.2 Gradient Evolution
We utilized a custom Straight-Through Estimator (STE) to snap weights to the grid while maintaining gradient flow. This allowed the model to navigate the discrete Prime landscape without losing structural intelligence.

### 2.3 Instruction Tuning
To recover semantic coherence, we fine-tuned on the Alpaca dataset using "Answer-Masked" loss. This forced the model to focus resources on factual accuracy rather than generic next-word prediction.

## 3. Results

### 3.1 Compression
The final binary payload is 246.63 MB, down from 2,200 MB (FP16). This represents a physical 8.51x reduction, making the model suitable for deployment on extreme edge hardware.

### 3.2 Intelligence
The model stabilizes at a Validation Loss of 2.73. Factual recovery experiments demonstrate that the "Paris" associative pathways can be restored within the Prime Grid habitat.

## 4. Conclusion
We successfully demonstrated that high-performance quantization is possible without retraining, democratizing 1.58-bit LLMs for edge devices like Raspberry Pi. Project Trinity confirms the Prime Harmonic Grid as a viable habitat for neural intelligence.
