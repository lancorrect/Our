## Introduction

It wil be my first paper in my whole life, but paper's name has not been decided. I will give it a nice name later.🤣

What I implement is based on **[DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA)**. Currently, the experiment has achieved *state-of-the-art* results on **4 out of 6 tasks**. I hope my model will outperform the latest SOTA results as soon as possible!

The following is my **expriment results**.

| **Models**   |  Restaurant  |  Restaurant  |    Laptop    |    Laptop    |   Twitter    |   Twitter    |
| ------------ | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
|              | **Accuracy** | **Macro-F1** | **Accuracy** | **Macro-F1** | **Accuracy** | **Macro-F1** |
| IAN          |    78.60     |      -       |    72.10     |      -       |      -       |      -       |
| RAM          |    80.23     |    70.80     |    74.49     |    71.35     |    69.36     |    67.30     |
| TNet         |    80.69     |    71.27     |    76.54     |    71.75     |    74.90     |    73.60     |
| ASGCN        |    80.77     |    72.02     |    75.55     |    71.05     |    72.15     |    70.40     |
| CDT          |    82.30     |    74.02     |    77.19     |    72.99     |    74.66     |    73.66     |
| TD-GAT       |     81.2     |      -       |     74.0     |      -       |      -       |      -       |
| BiGCN        |    81.97     |    73.48     |    74.59     |    71.84     |    74.16     |    73.35     |
| kumaGCN      |    81.43     |    73.64     |    76.12     |    72.42     |    72.45     |    70.77     |
| R-GAT        |    83.30     |    76.08     |    77.42     |    73.76     |    75.57     |    73.82     |
| DGEDT        |    83.90     |    75.10     |    76.80     |    72.30     |    74.80     |    73.40     |
| DualGCN      |    84.27     |    78.08     |    78.48     |    74.74     |    75.92     |    74.29     |
| SSEGCN       |    84.72     |    77.51     |  **79.43**   |  **76.49**   |  **76.51**   |  **75.32**   |
| **Our**      |  **84.90**   |  **78.29**   |    78.48     |    75.32     |  **76.51**   |    74.54     |
|              |              |              |              |              |              |              |
| BERT         |    85.97     |    80.09     |    79.91     |    76.00     |    75.92     |    75.18     |
| R-GAT+BERT   |    86.60     |    81.35     |    78.21     |    74.07     |    76.15     |    74.88     |
| DGEDT+BERT   |    86.30     |    80.00     |    79.80     |    75.60     |    77.90     |    75.40     |
| BERT4GCN     |    84.75     |    77.11     |    77.49     |    73.01     |    74.73     |    73.76     |
| T-GCN+BERT   |    86.16     |    79.95     |    80.88     |    77.03     |    76.45     |    75.25     |
| DualGCN+BERT |    87.13     |    81.16     |    81.80     |    78.10     |    77.40     |    76.02     |
| SSEGCN+BERT  |    87.31     |    81.09     |    81.01     |    77.96     |    77.40     |    76.02     |
| **Our+BERT** |  **88.03**   |  **82.84**   |  **82.28**   |  **78.80**   |  **78.14**   |  **76.65**   |

### Come on man!

