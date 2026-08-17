# Experimental Configuration for the ARFIU Method

## 1. Dataset Split

**Table 1: Static dataset split**

| Dataset        | Training set sample size | Test set sample size |
|----------------|--------------------------|----------------------|
| Adult Income   | 30162                    | 15060                |
| Compas         | 4812                     | 2402                 |
| Synthetic      | 30162                    | 15060                |

**Table 2: Online dataset split 1** (NYSF dataset, 5 boroughs, 15 tasks)

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th colspan="3">BROOKLYN (Domain 1)</th>
<th colspan="3">QUEENS (Domain 2)</th>
<th colspan="3">MANHATTAN (Domain 3)</th>
<th colspan="3">BRONX (Domain 4)</th>
<th colspan="3">STATEN IS (Domain 5)</th>
</tr>
<tr>
<th>Task 1</th><th>Task 2</th><th>Task 3</th>
<th>Task 4</th><th>Task 5</th><th>Task 6</th>
<th>Task 7</th><th>Task 8</th><th>Task 9</th>
<th>Task 10</th><th>Task 11</th><th>Task 12</th>
<th>Task 13</th><th>Task 14</th><th>Task 15</th>
</tr>
</thead>
<tbody>
<tr>
<td>NYSF</td>
<td>15522</td><td>19433</td><td>18727</td>
<td>11723</td><td>13411</td><td>12405</td>
<td>12650</td><td>10150</td><td>10863</td>
<td>11532</td><td>9833</td><td>10665</td>
<td>2337</td><td>2639</td><td>1922</td>
</tr>
</tbody>
</table>

**Table 3: Online dataset split 2** (Credit dataset, 3 domains, 6 tasks)

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th colspan="2">Domain 1</th>
<th colspan="2">Domain 2</th>
<th colspan="2">Domain 3</th>
</tr>
<tr>
<th>Task 1</th><th>Task 2</th>
<th>Task 3</th><th>Task 4</th>
<th>Task 5</th><th>Task 6</th>
</tr>
</thead>
<tbody>
<tr>
<td>Credit</td>
<td>500</td><td>500</td>
<td>500</td><td>500</td>
<td>500</td><td>500</td>
</tr>
</tbody>
</table>

The Adult Income dataset contains 48,842 samples; after removing missing data (samples containing "?"), we obtain a clean dataset containing 45,222 samples. For the NYSF dataset, data from January, May, and October in each borough were selected to serve as the complete dataset for a single task.

Links to some official datasets:

1. https://archive.ics.uci.edu/ml/datasets/adult  
2. https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis  
3. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

---

## 2. Hyperparameter Settings

**Table 4: Hyperparameter settings on static datasets**

| Datasets         | Adult Income          | Compas                | Synthetic                          |
|------------------|-----------------------|-----------------------|------------------------------------|
| *epoch*          | 200                   | 200                   | 200                                |
| *λ*              | 1                     | 1                     | 1                                  |
| *δ* (DP/EOD/EOP) | 1e2 / 1e1 / 1e2       | 8e6 / 7e6 / 1e3       | 5e8 / 1e31 (1e0/1e1)               |
| *η* (DP/EOD/EOP) | 3.1 / 3.9 / 4.9       | 1 / 2 / 2             | 0.9 / 3.1 / 4.5                    |

**Table 5: Hyperparameter settings on online datasets**

| Datasets         | NYSF dataset          | Credit dataset        |
|------------------|-----------------------|-----------------------|
| *epoch*          | 30                    | 30                    |
| *batchsize*      | 4096                  | 500                   |
| *learning rate*  | 1e-3                  | 1e-3                  |
| *λ*              | 1                     | 1                     |
| *δ*              | 1                     | 1                     |
| *η*              | 1                     | 2                     |

---

## Implementation and Hyperparameter Settings of Baseline Methods

### 1. Static Datasets

(1) For the **Reweighing** (Kamiran & Calders, 2012), **AD** (Zhang et al., 2018), **EO** (Hardt et al., 2016), and **CEO** (Pleiss et al., 2017), these methods are primarily implemented using the open-source toolkits IBM AIF360 (Bellamy et al., 2018) and Fairlearn (Bird et al., 2020); for specific usage instructions, please refer to the official examples.

(2) For the **Fairbatch** (Roh et al., 2020), **FairMixup** (Chuang & Mroueh, 2021), **LBC** (Jiang & Nachum, 2020), **FC** (Zafar et al., 2017), **RFI** (Baharlouei et al., 2019), and **APW** (Hu et al., 2025), we primarily relied on the official code provided by the authors for implementation. The relevant code links are as follows:

- FC: https://github.com/mbilalzafar/fair-classification  
- LBC: https://github.com/google-research/google-research/tree/master/label_bias  
- RFI: https://github.com/optimization-for-data-driven-science/Renyi-Fair-Inference  
- FairMixup: https://github.com/chingyaoc/fair-mixup  
- Fairbatch: https://github.com/yuji-roh/fairbatch  
- APW: https://github.com/che2198/APW  

The hyperparameter configurations for each method are described below; the specific values still need to be determined through tuning under practical operating conditions:

#### FC
- **Classifier**: LogisticRegression  
- **max_iter** (max iterations): recommended range 50–200, recommended value 100  
- **c** (fairness constraint threshold): recommended ranges 0.01–0.3 (for DP metric) and 0.01–0.15 (for EOP/EOD metrics)  
- **tol** (convergence tolerance): recommended value 1e-4

#### LBC
- **Classifier**: LogisticRegression  
- **max_iter**: recommended range 20–50, recommended value 50  
- **η** (weight update step size): recommended ranges 1.0–3.0 (for DP) and 0.001–0.02 (for EOP/EOD)  
- **tol**: recommended range 1e-4 to 1e-3, recommended value 1e-4

#### RFI
- **Classifier**: LogisticRegression  
- **max_iter**: recommended range 50–200, recommended value 50  
- **λ** (fairness penalty coefficient): from {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 6.0, 8.0, 10.0}, recommended value 1.0  
- **α** (Rényi order): recommended range 1–16  
- **tol**: recommended range 1e-6–1e-3, recommended value 1e-4

#### FairMixup
- **Classifier network architecture**: uses the official code  
- **batchsize**: recommended 1,000 (Adult), 200 (Compas), 2,000 (Synthetic)  
- **epochs**: recommended 200  
- **learning rate**: recommended 1e-3  
- **α**: recommended value 1.0  
- **λ**: recommended range for DP/EOD: 0.1–0.7; for EOP: 0.5–5

#### Fairbatch
- **Classifier network architecture**: uses the official code  
- **batchsize**: recommended 1,000 (Adult), 200 (Compas), 2,000 (Synthetic)  
- **epochs**: recommended {140, 200, 250}  
- **learning rate**: recommended 1e-2 or 1e-3  
- **α** (adjustment step size): recommended range 0.0001–0.1, with recommended values from {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5}

#### APW
- **Classifier**: LogisticRegression  
- **epochs**: recommended 200  
- **α** (step size): recommended range 0.5–5  
- **subgroup learning rate**: recommended from {1e1, 1e2, 1e3, 1e4, 1e6, 3e6, 4e6, 5e6, 1e7, 1e8, 3e8, 4e8, 1e9}  
- **d** (decision boundary): recommended value 0.5

---

### 2. Online Datasets

For **FairDolce** (Zhao et al., 2023), **FairSAOML** (Zhao et al., 2024), **LTFconVAE** (Chen et al., 2025), and **ConSupFairOL** (Chen et al., 2026), we employ the official code provided by the authors or implement them based on the algorithms described in the papers. The hyperparameter tuning settings for each method are described as follows:

#### FairDolce (https://github.com/harderbetter/fairdolce)
- **batchsize**: recommended 4096 (NYSF) and 500 (Credit)  
- **epochs**: recommended 30  
- **Initial dual parameters** (*λ₁*, *λ₂*, *λ₃*): recommended (0.5, 0.8, 0.5)  
- **Learning rates** *n₁* and *n₂*: recommended 1e-3 and 1e-2  
- **Boundary values** (*ε₁*, *ε₂*, *ε₃*): recommended (0.025, 0.025, 0.01)

#### LTFconVAE
- **batchsize**: recommended 4096 (NYSF) and 500 (Credit)  
- **epochs**: recommended 30  
- **Initial dual parameters** (*w₁*, *w₂*, *w₃*): recommended (0.5, 0.8, 0.0)  
- **Learning rates** *n₁* and *n₂*: recommended 1e-3 and 1e-2  
- **Boundary values** (*m₁*, *m₂*, *m₃*): recommended (0.025, 0.025, 0.5)

#### ConSupFairOL
- **batchsize**: recommended 4096 (NYSF) and 500 (Credit)  
- **epochs**: recommended 30  
- **Initial dual parameter** *w*: recommended 0.1  
- **Learning rates** *n₁* and *n₂*: recommended 1e-3 and 1e-2  
- **Boundary parameter** *ρ*: recommended 0.025  
- **Temperature coefficient** *τ*: recommended 0.7 (NYSF) and 0.8 (Credit)  
- **λ**: recommended 0.2 (NYSF) and 0.1 (Credit)

#### Suggested parameter tuning for the above methods
- **Initial range for dual parameters**: {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 10, 50, 100}  
- **Learning rates *n₁* and *n₂***: {0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}  
- **Boundary values**: {0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06}

#### FairSAOML
- **batchsize**: recommended 800 (NYSF) and 500 (Credit)  
- **Initial dual parameter** *λ*: recommended range {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000}; recommended value: 0.1  
- **Learning rates *n₁* and *n₂***: recommended range {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000}; recommended value: 1e-3  
- **Number of iterations *N***: recommended range {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100}, recommended value 30

#### Network architectures of the baseline methods
The semantic/variable encoders and decoders of FairDolce and LTFconVAE, the encoder of ConSupFairOL, and the feature extractor of the proposed method share the same network architecture, each consisting of one linear layer followed by a LeakyReLU activation function and a Batch Normalization layer; meanwhile, the classifiers of all methods are the same, comprising Batch Normalization, a Sigmoid activation function, and one linear layer.
