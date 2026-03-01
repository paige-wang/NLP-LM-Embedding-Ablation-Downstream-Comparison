ST5230: Applied Natural Language Processing
Assignment 1

This assignment covers Lectures 1-4, including language modeling, word embeddings, RNN/LSTM models, and Transformers.

General Requirements:
- You must choose your own English text dataset (any domain).
- All required models must be trained by yourself.
- You may use PyTorch, TensorFlow, and HuggingFace.
- Model size should be computationally reasonable.

Part I: Language Model Training and Comparison (40 points)
Using the same dataset, train the following language models:
- n-gram language model
- RNN language model
- LSTM language model
- Transformer language model

Compare these models in terms of training time, inference time, quantitative performance (e.g., perplexity or loss), and qualitative behavior such as generated text samples. Summarize the main differences you observe. You may choose model architectures and hyperparameters as appropriate.

Part II: Embedding Variants and Ablation (30 points)
Modify the embedding layer in your neural language models. You should consider at least the following settings:
- Trainable embeddings learned from scratch.
- Fixed embeddings trained by yourself (e.g., Word2Vec or GloVe-style).
- Fixed pretrained embeddings from public sources (e.g., Hugging Face).

Compare model behavior under different embedding choices, focusing on performance, training stability, and convergence speed.

Part III: Downstream Task with Learned Representations (30 points)
Using representations from one trained language model, build a simple downstream task such as sentiment analysis. You may choose to freeze or fine-tune the language model, but the downstream model itself should remain simple. Report downstream performance and briefly explain which representations work better and how this relates to differences between language models.

Submission:
- Submit a PDF report (maximum 6 pages) and your code (either as a GitHub link or a zip file).
- The report should focus on results and comparisons; unnecessary background will not be rewarded.
- The report should clearly state: the dataset used, the models and variants compared, the evaluation criteria, and the main observations.
- Comparable settings do not require identical architectures or budgets, but comparisons should be reasonable and clearly explained.

Grading Guidelines: We want High Credit
- Basic Credit (Pass / B range): All required models are implemented and trained, comparisons are conducted under comparable settings, and at least one quantitative and one qualitative result are reported clearly.
- High Credit (A range): Awarded for careful experimental control, insightful comparisons across model families, meaningful embedding ablations, and clear interpretation of results in terms of model architecture.
- Point Deductions: Missing required models, unfair or unclear comparisons, reporting numbers without explanation, or excessive unrelated content.

Note:
There are no preset requirements on dataset size, model size, specific architectures, or evaluation metrics. Grades will be based on the quality of comparisons and explanations.