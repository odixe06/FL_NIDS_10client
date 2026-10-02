# A novel federated learning approach for IoT botnet intrusion detection using SHAP-based knowledge distillation

**Md. Alamgir Hossain** \(^{1,2}\) · **Sadman Saif** \(^{1}\) · **Md. Saiful Islam** \(^{1}\)

Received: 22 November 2024 / Accepted: 17 June 2025 / Published online: 21 August 2025  
© The Author(s) 2025

## Abstract

The exponential growth of the Internet of Things (IoT) has introduced new security vulnerabilities, particularly from botnet attacks that exploit the heterogeneity and limited processing capabilities of IoT devices. Traditional centralized intrusion detection models are ineffective in protecting distributed IoT environments due to data privacy concerns and the challenges posed by non-IID (non-independent and identically distributed) data. In response, we propose a novel, privacy-preserving federated learning framework tailored for IoT intrusion detection. Our framework leverages SHAP (Shapley Additive Explanations), a technique for computing feature importance, to provide interpretable insights while maintaining data privacy. Each IoT client trains locally on its unique, heterogeneous data, computes SHAP values to quantify feature relevance, and shares only distilled feature knowledge with the central server. This aggregated knowledge forms a global feature profile that enables the global model to accurately detect diverse botnet intrusions across non-IID client data. Experimental results demonstrate that our model achieves near-perfect accuracy (99.99%) across various botnet types, showcasing robustness in identifying botnet-specific attack patterns while preserving privacy. By addressing IoT data heterogeneity, non-IID data, and privacy concerns, our framework provides a scalable, interpretable, and privacy-compliant federated learning solution, advancing the security of IoT networks against botnet intrusions.

**Keywords** Federated IoT security · Privacy-preserving intrusion detection · SHAP-based knowledge distillation · Decentralized feature aggregation · Non-IID data handling · Botnet detection in IoT · Distributed IoT security framework · Heterogeneous IoT networks

---

## Introduction

### Background and motivation

Botnets such as Bashlite, Mirai, Neris, RBoT, and Torii [1-3] are among the most pervasive threats to the IoT landscape, exploiting vulnerabilities in low-security, resource-limited devices to launch extensive and highly disruptive cyber-attacks. These botnets hijack IoT devices to form vast networks under remote control, which are weaponized for destructive activities like DDoS attacks that disrupt critical services, data exfiltration for unauthorized access, and even manipulation of IoT networks for large-scale data breaches [4-6]. In an IoT environment characterized by diverse devices, from smart home products to industrial systems, the heterogeneity of hardware, software, and network protocols compounds the complexity of detecting such attacks. This heterogeneity, coupled with non-IID data where data distributions vary widely between devices, creates formidable barriers for intrusion detection systems.

Addressing botnet-driven disruptions in IoT environments is essential due to the increasing reliance on IoT devices in critical sectors such as healthcare, finance, transportation, and smart cities. Botnets have demonstrated their potential to cause widespread service outages, financial loss, and even physical harm by exploiting vulnerabilities in IoT systems.

---

### Table 1: Short forms and corresponding full terms used in the research paper

| Short form | Full term |
|------------|-----------|
| Non-IID | Non-independent and identically distributed |
| SHAP | Shapley additive explanations |
| FL | Federated learning |
| IoT | Internet of things |
| DDoS | Distributed denial of service |
| FDL | Federated deep learning |
| SGDC | Stochastic gradient descent classifier |
| IHHO-NN | Harris hawks optimization algorithm for neural networks |
| ER-VEC | Extra tree based random—voting ensemble classifier |
| RF | Random forest |
| GWO | Gray wolves optimization |
| LGBA-NN | Local–global best bat algorithm for neural networks |
| CNN-LSTM | Convolutional neural network with a long short-term memory |
| AU-ANN | Deep autoencoder with artificial neural network |

---

## Literature review

The rapid evolution of IoT networks has driven extensive research in intrusion detection systems (IDS), with a particular focus on securing these networks against botnet attacks. In this section, we review relevant literature on IoT botnet detection, federated learning, privacy-preserving techniques, and SHAP-based interpretability methods. These studies provide foundational insights that informed the development of our novel federated learning framework.

### IoT botnet detection approaches

IoT botnet attacks exploit vulnerabilities in resource-constrained IoT devices, highlighting the limitations of traditional machine learning (ML) and deep learning (DL) classifiers in intrusion detection [15-18]. While centralized models such as Support Vector Machines (SVM), Decision Trees, and Neural Networks have been widely used to analyze network traffic for botnet detection, they face significant challenges in real-world IoT environments. These models struggle to handle evolving and rapidly changing botnet attack architectures, as they are not designed for adaptive, distributed detection. Furthermore, centralized approaches are ill-suited for IoT due to data privacy constraints and their inability to manage non-IID data across heterogeneous devices [19-21].

### Federated learning for IoT security

Federated learning (FL) has emerged as a promising solution to IoT security challenges, allowing data to remain decentralized across devices while still contributing to a global model [22]. Research has shown that FL can maintain privacy by enabling clients to train locally on their data and share model updates instead of raw data.

Regan et al. [23] proposes a federated learning approach using a deep autoencoder for IoT botnet detection, emphasizing data privacy by processing data locally on devices. The model learns normal device behaviors and flags deviations as potential threats, achieving up to 98% accuracy in detecting botnet attacks like Mirai and Bashlite. While effective in preserving data privacy, this approach faces challenges with communication overhead and limited adaptability to complex, evolving attack patterns. Sayan and Hanawal [24] presents a federated learning approach using a hybrid ensemble model, Probabilistic Hybrid Ensemble Classification (PHEC), for IoT intrusion detection. In a centralized setup, PHEC combines K-Nearest Neighbors and Random Forest classifiers to maximize the True Positive Rate (TPR) while minimizing the False Positive Rate (FPR). The federated version, FedStacking, applies Multi-layer Perceptrons (MLPs) on distributed clients to preserve data privacy.

Despite achieving strong detection rates, the approach faces significant limitations: handling non-IID data in federated setups impacts model aggregation and reduces TPR on rare attacks, and using simplified MLP models in the federated setting limits flexibility, making it challenging to maintain low FPR across heterogeneous data distributions.

Popoola et al. [25] proposed an FDL approach for botnet detection on the Bot-IoT and N-BaIoT datasets, achieving up to 98.6% accuracy. The model effectively handled data privacy constraints and was optimized for resource-limited IoT devices, showing robust performance in detecting common botnet attacks. However, despite these positive results, the FDL model encountered limitations in detecting minority classes and zero-day attacks, with variability in TPR and an increase in FPR for underrepresented classes. Additionally, the federated training process required significant computational resources, leading to longer training times, particularly challenging in real-time IoT environments. Another approach proposed by Zhang et al. [26] for detecting zero-day botnet attacks in IoT networks, utilizing a K-greedy aggregation algorithm to prioritize model updates from devices with higher detection uncertainty. This strategy enhances the global model's adaptability to unknown attack patterns while maintaining data privacy by training locally on IoT devices. Tested on the N-BaIoT dataset, the approach achieved up to 92.42% accuracy with a low false alarm rate of 0.48%, significantly outperforming traditional FL methods like FedAvg, which averaged 59.87% accuracy. However, the framework faces challenges with fluctuating performance across devices and computational demands due to reliance on uncertainty estimation.

Zhang et al. [27] explore a FL framework for IIoT intrusion detection, incorporating instance-based transfer learning to address non-IID data and imbalance issues. Using a weighted voting aggregation method based on maximum mean discrepancy (MMD), the approach applies AdaBoost and Random Forest as base models. Tested on a physical IIoT testbed with Raspberry Pis, the model achieved promising accuracy, reaching 95.97% with AdaBoost. However, limitations include fluctuating accuracy due to data heterogeneity, potential scalability issues with ensemble models, and challenges in asynchronous network scenarios, affecting its applicability in dynamic real-world IIoT environments.

Sarhan et al. [28] propose a FL-based cyber threat intelligence (CTI) sharing scheme to enhance network intrusion detection systems (NIDS) across multiple organizations without data sharing. They leverage federated learning to enable each organization to train models on local data, thereby preserving privacy. The methodology includes initiating a global model and averaging locally trained model updates to produce an aggregated model capable of detecting varied network intrusions. The evaluation compares federated, centralized, and localized learning approaches using two diverse datasets, NF-UNSW-NB15-v2 and NF-BoT-IoT-v2. Results indicate that federated learning achieves detection rates close to centralized models, particularly excelling in non-IID environments, while maintaining data privacy. However, limitations include higher false alarm rates (FAR) compared to centralized methods, and challenges with detection in highly heterogeneous data, underscoring the need for improved adaptability in federated models.

### Non-IID data challenges in federated learning for IoT

IoT datasets are often non-IID, with client data exhibiting variability due to device-specific factors and usage patterns. Federated learning in such settings can struggle to generalize due to data imbalance across clients. Previous studies have addressed this by implementing techniques like transfer learning and weighted aggregation to improve model accuracy across heterogeneous data sources. However, these solutions often do not integrate interpretation to discern how well the model adapts to client-specific patterns [29, 30]. However, the non-IID nature of IoT data, which varies across devices and networks, remains a critical challenge. Many studies have incorporated weighted aggregation techniques and transfer learning within FL frameworks to address these data heterogeneity issues, though without fully tackling interpretability and privacy concerns.

Other approaches to IoT botnet detection, summarized in Table 2, highlight significant limitations in centralized methods when addressing IoT-specific challenges. Centralized models, such as DNN [31], A³E [32], and SGDC [33], demonstrate high performance on individual datasets (e.g., N-BaIoT, MedBIoT, NCC2) but fail to address critical issues like data heterogeneity and non-IID distributions. While methods like IHHO-NN [34] and ER-VEC [35] incorporate dimensionality reduction, they still rely on centralized data collection, which compromises scalability and raises privacy concerns. The inability of these approaches to handle heterogeneous IoT environments or adapt to non-IID data significantly limits their applicability in real-world scenarios. Moreover, none of the reviewed methods incorporate explainable artificial intelligence (XAI) techniques, leaving them as black-box solutions with limited interpretability for network administrators. Despite the inclusion of advanced architectures like CNN-LSTM [36] and Bi-LSTM [37], the lack of focus on addressing data imbalance, resource constraints, and decentralized learning frameworks creates a substantial gap in the literature. This gap underscores the urgent need for federated and decentralized approaches that can effectively handle IoT-specific challenges, ensure privacy, and provide explainable insights for robust botnet detection.

---

### Summary of gaps and our approach

Existing federated learning approaches for IoT intrusion detection encounter several limitations when addressing the diverse and dynamic nature of IoT environments. Most traditional methods struggle with data heterogeneity and feature misalignment across client devices, making it challenging to achieve effective model generalization in non-IID settings. Additionally, client heterogeneity, including variations in computational resources and data quality, further complicates model consistency and convergence. While some studies have applied transfer learning or ensemble methods to handle these variances, they often face challenges with computational overhead, high false positive rates, and reduced adaptability to new or evolving botnet threats. Moreover, many approaches lack interpretability, which is critical for understanding attack patterns and enhancing security in distributed IoT networks.

To address these gaps, our proposed framework introduces a privacy-preserving federated learning model that integrates SHAP-based feature importance aggregation for improved interpretability and model alignment across clients. By enabling each IoT client to train locally on its heterogeneous data and contribute distilled feature insights, our approach overcomes data and feature heterogeneity. We also optimize model performance for devices with limited resources, ensuring scalable adaptation across varying client capacities. Our contributions include a decentralized feature aggregation strategy to handle non-IID data distributions effectively, enhanced model interpretability through SHAP values, and robust intrusion detection across multiple botnet types, addressing the critical limitations of existing FL models for IoT security. This work establishes a scalable, interpretable, and privacy-compliant federated learning framework for robust, real-world IoT intrusion detection.

---

## Methodology

This section describes the methodology of the proposed research, detailing the entire process from data preparation to federated model evaluation and analysis. The overall pipeline of the development approach is illustrated in Fig. 1.

### Dataset preparation

Dataset preparation involves selecting and organizing relevant IoT botnet datasets to create a diverse and representative environment for model training and evaluation. In this research, three widely recognized datasets MedBIoT, N-BaIoT, and NCC2 are used to simulate various botnet attacks and normal traffic, providing a realistic foundation for federated learning with non-IID data distributions across clients.

#### MedBIoT dataset

The MedBIoT dataset [44] is designed for IoT botnet detection and combines data from real and emulated IoT devices within a medium-sized network of 83 devices. This dataset includes actual malware traffic generated by prominent botnets such as Mirai, Bashlite, and Torii, as well as normal traffic. The data is labeled, allowing for easy classification and analysis of normal and malware activity. With 101 initial features, this dataset offers a detailed foundation for analyzing botnet behaviors, particularly in early stages like spreading and C&C communication.

#### N-BaIoT dataset

The N-BaIoT dataset [50], was developed to address the challenge of detecting IoT-based botnet attacks by leveraging network-based anomaly detection. It includes traffic data from nine commercial IoT devices infected with Mirai and Bashlite botnets, two of the most prevalent IoT-based malware strains. These devices generated real-world traffic, allowing for authentic testing scenarios involving common botnet operations, including DDoS and scanning attacks. We employed the N-BaIoT dataset, which initially comprises 115 attributes. This dataset proved particularly suitable for training and evaluating our federated learning approach due to its comprehensive representation of real and varied IoT traffic patterns.

#### NCC2 dataset

The NCC2 dataset [3] simulates complex botnet attack scenarios by integrating botnet activity from the CTU-13 and NCC datasets. It contains data in the form of bidirectional network flows, or "bientflow" files, with 18 features representing network headers. Unlike traditional datasets that focus on sporadic or periodic attacks, the NCC2 dataset includes simultaneous botnet activities, reflecting intense, concurrent attack behaviors. This advanced characteristic models real-world scenarios where security systems must manage multiple attack vectors in short time intervals, making it particularly valuable for analyzing the robustness of intrusion detection systems. The dataset includes both normal and malicious activity, with notable botnets such as Neris and RBot, providing a comprehensive testing ground for federated learning in IoT botnet detection.

### Data preprocessing

Data preprocessing is a crucial first step in the proposed methodology to ensure data consistency, integrity, and compatibility with the machine learning models. In this research, data preprocessing includes handling missing values, removing duplicates, and normalizing and encoding features to prepare the dataset for subsequent stages.

#### Handling missing values

The initial dataset often contains inconsistencies, such as infinity values and extremely large numbers that could distort model performance. To address this, all instances of infinity values and unusually large numeric values are replaced with `NaN`. Formally, if a dataset \(D\) contains features \(X = \{x_1, x_2, \ldots , x_n\}\), then:

\[X = \{x_i \mid x_i \neq \infty \text{ and } x_i < \text{threshold}\} \quad (1)\]

where any \(x_i\) that meets the conditions for infinity or exceeds a set threshold is set to `NaN`.

After handling these values, all rows containing `NaN` are dropped to maintain data integrity:

\[D = D \setminus \{\text{rows containing NaNs}\} \quad (2)\]

#### Duplicate removal

Duplicate records in the dataset can lead to biased model performance and redundant computations. We remove duplicate rows by identifying and discarding any records that are identical across all feature values. Let \(D = \{d_1, d_2, \ldots, d_m\}\) represent the dataset where each \(d_i\) is a unique data point. Duplicates are identified as follows:

\[D = D \setminus \{\text{duplicate rows in } D\} \quad (3)\]

This step ensures that each data point in \(D\) is unique, reducing redundancy and computational overhead.

#### Normalization and encoding

To make the data more suitable for model training, we apply normalization to numerical features and encoding to categorical features. These steps enhance the compatibility of the data with machine learning models, ensuring that numerical values are scaled consistently and categorical variables are transformed into numeric form.

**Numerical feature scaling:** Numerical features often have different scales and ranges, which can lead to biased model training if not normalized. We standardize all numerical columns using the StandardScaler, which transforms each feature \(x_i\) in the numerical feature set \(X\) to have a mean of 0 and a standard deviation of 1. For each numerical feature \(x_i\) in dataset \(D\), normalization is applied as:

\[x_i' = \frac{x_i - \mu_{x_i}}{\sigma_{x_i}} \quad (4)\]

where \(\mu_{x_i}\) and \(\sigma_{x_i}\) are the mean and standard deviation of \(x_i\) respectively. This transformation ensures that all numerical features contribute equally to the model, preventing any single feature from disproportionately influencing the results.

**Categorical encoding:** Many features in the dataset are categorical and need to be transformed into numeric values to be compatible with the machine learning models. Using LabelEncoder, each unique category in a feature \(C = \{c_1, c_2, \ldots, c_k\}\) is assigned a unique integer label. For each categorical feature \(c\) in \(C\), encoding is defined as:

\[c \rightarrow \text{integer label} \quad (5)\]

This process effectively maps each category in \(C\) to a unique numeric label, allowing categorical information to be seamlessly incorporated into the model.

By handling missing values, removing duplicates, normalizing numerical features, and encoding categorical features, this preprocessing stage ensures that the data is clean, consistent, and ready for the feature selection and federated learning steps that follow.

### Feature selection

The feature selection process refines the dataset by identifying and retaining only the most relevant features, which improves model accuracy and reduces computational complexity. This research employs a multi-step approach to feature selection, combining correlation analysis, mutual information, and Principal Component Analysis (PCA) to ensure the final feature set is both relevant and efficient.

#### Correlation analysis

Correlation analysis identifies highly correlated features to reduce redundancy and multicollinearity, which can negatively affect the model's performance. We calculate the correlation matrix \(C\) for the feature set \(X\), with absolute correlation values \(|r_{i,j}|\) representing the degree of correlation between features \(x_i\) and \(x_j\). Features with \(|r_{i,j}| > 0.8\) are flagged for further evaluation. To ensure that each retained feature provides unique and relevant information about the target variable, we prioritize features based on their importance to the target variable. Features with correlations above this threshold are retained:

\[X_{corr} = \{x_i \in X \mid |Corr(x_i, x_j)| > 0.8\} \quad (6)\]

This ensures that only strongly correlated features that provide unique information about the target variable are retained.

#### Mutual information analysis

Mutual information measures the dependency between each feature and the target variable \(y\), helping identify features that contribute the most to target prediction. Using the mutual_info_classif function, we compute the mutual information \(I(x_i, y)\) for each feature \(x_i\) with respect to \(y\). Features are sorted based on their mutual information values, and those contributing up to 95% of the cumulative mutual information are selected. Let \(I_{sorted}\) be the sorted mutual information values, and \(CumSum(I_{sorted})\) be the cumulative sum:

\[k = \min\left\{j \mid \frac{CumSum(I_{sorted})[j]}{CumSum(I_{sorted})[-1]} \geq 0.95\right\} \quad (7)\]

\[X_{MI} = \{x_i \in X \mid i \leq k\} \quad (8)\]

This step ensures that the final set includes only features that capture significant predictive information about the target variable.

#### Principal component analysis (PCA)

PCA is a dimensionality reduction technique that identifies orthogonal components in the data, capturing the maximum variance in a reduced feature set. We perform PCA on the feature set \(X\) to calculate principal components, retaining those that explain 95% of the total variance. Let \(\lambda_i\) denote the explained variance by component \(i\). The number of components \(n\) retained is given by:

\[n = \min\left\{j \left| \frac{\sum_{i=1}^{j}\lambda_i}{\sum_{i=1}^{m}\lambda_i} \geq 0.95\right.\right\} \quad (9)\]

The features associated with these top \(n\) components are selected as \(X_{PCA}\). This step reduces feature dimensionality while preserving the majority of the information in the original data.

#### Intersection of selected features

To finalize the feature set, we take the intersection of features selected from the three methods: correlation analysis, mutual information, and PCA. The final feature set \(X_{selected}\) is given by:

\[X_{selected} = X_{corr} \cap X_{MI} \cap X_{PCA} \quad (10)\]

By focusing on the features that are common across all three methods, we obtain a refined feature set that balances relevance, diversity, and information density. This final selection step ensures that the dataset is optimized for effective model training and efficient computation in the federated learning process. The selected features with their importance score for the different datasets are presented in Fig. 2.

### Client data partitioning for federated learning

To implement a federated learning framework that reflects real-world IoT environments, we partition the dataset across multiple clients, each receiving a subset of data with a unique combination of botnet types. This setup creates non-IID (non-independent and identically distributed) data distributions, simulating the heterogeneity common in decentralized IoT networks. Multiple partitioning settings are defined to test the model's adaptability and performance across different configurations.

#### Data partitioning based on botnet types

Each client is assigned data specific to particular botnet types, creating variability in the distribution and class composition for each client. This ensures that the clients' datasets are non-IID, making the learning process more representative of real-world IoT scenarios where devices may be exposed to different botnet types. Let \(D\) represent the full dataset, and \(D_k \subset D\) be the subset of data assigned to client \(k\). The partitioning based on botnet types is as follows:

\[D_k = \{(X_i, y_i) \mid y_i \in \text{botnet types assigned to client } k\} \quad (11)\]

For instance, if client 1 is assigned Bashlite and Normal data, then:

\[D_1 = \{(X_i, y_i) \mid y_i \in \{\text{Bashlite}, \text{Normal}\}\} \quad (12)\]

This non-IID partitioning approach allows each client to train on a subset of the total dataset that reflects specific types of botnet attacks, introducing heterogeneity and simulating a distributed IoT network under various attack patterns.

#### Define Client datasets for multiple settings

To further explore the performance and adaptability of the federated learning model, we define four unique settings, each with different client configurations and combinations of botnet types from different datasets.

**Setting 1:** Clients receive data containing combinations of the most prevalent botnet types: Bashlite, Mirai, Torii, and Normal from MedBIoT dataset. This setting provides a balanced distribution across clients, with each client handling two or more botnet types. The data distribution for this setting is as follows: \(D1 = \{Bashlite, Normal\}\), \(D2 = \{Mirai, Normal\}\), \(D3 = \{Torii, Normal\}\).

**Setting 2:** Clients are assigned combinations of botnet types with an emphasis on minimal class overlap. This setup isolates botnet types across specific clients, allowing for the evaluation of the model's performance on less complex and more distinct data configurations. The data distribution for this setting from the dataset MedBIoT is: \(D1 = \{Bashlite, Normal\}\), \(D2 = \{Mirai, Normal\}\), \(D3 = \{Torii, Normal\}\), \(D4 = \{Bashlite, Mirai, Normal\}\), \(D5 = \{Bashlite, Torii, Normal\}\), \(D6 = \{Mirai, Torii, Normal\}\), \(D7 = \{Bashlite, Mirai, Torii, Normal\}\).

**Setting 3:** This setting introduces additional botnet types, such as RBoT and Neris, alongside Bashlite, Mirai, and Torii from MedBIoT, N-BaIoT, NCC2 datasets. Clients receive data with a broader combination of botnet types and benign traffic, increasing the diversity of data distribution and enabling the evaluation of the model's adaptability to more complex, mixed attack scenarios. The data distribution is as follows: \(D1 = \{Bashlite, Normal\}\), \(D2 = \{Mirai, Normal\}\), \(D3 = \{Torii, Normal\}\), \(D4 = \{Neris, Normal\}\), \(D5 = \{RBoT, Normal\}\).

Data distribution across clients with various types of botnet attacks and normal for different settings is shown in Fig. 3. And the amount of data for each client across different clients was used to develop the model presented in Table 3.

Each setting simulates different levels of client data diversity, enabling a comprehensive evaluation of the model's performance under varying levels of heterogeneity and non-IID data distributions. By partitioning data in these ways, the approach robustly captures the decentralized and diverse nature of IoT networks, enhancing the federated learning model's ability to generalize across distinct botnet attack patterns.

---

### Table 3: Amount of data for each client in different settings

| Setting | Client | Data distribution |
|---------|--------|-------------------|
| Setting 1 (MedBIoT dataset) | Client 1 | 111005 (Normal), 56649 (Bashlite) |
| | Client 2 | 111005 (Normal), 88989 (Mirai) |
| | Client 3 | 111005 (Normal), 96530 (Torii) |
| Setting 2 (MedBIoT dataset) | Client 1 | 111005 (Normal), 56649 (Bashlite) |
| | Client 2 | 111005 (Normal), 88989 (Mirai) |
| | Client 3 | 111005 (Normal), 96530 (Torii) |
| | Client 4 | 111005 (Normal), 88989 (Mirai), 56649 (Bashlite) |
| | Client 5 | 111005 (Normal), 96530 (Torii), 56649 (Bashlite) |
| | Client 6 | 111005 (Normal), 96530 (Torii), 88989 (Mirai) |
| | Client 7 | 111005 (Normal), 96530 (Torii), 88989 (Mirai), 56649 (Bashlite) |
| Setting 3 (MedBIoT, N-BaIoT, NCC2 datasets) | Client 1 | 304044 (Bashlite), 40395 (Normal) |
| | Client 2 | 652100 (Mirai), 40395 (Normal) |
| | Client 3 | 370042 (Normal), 321776 (Torii) |
| | Client 4 | 314000 (Neris), 103833 (Normal) |
| | Client 5 | 134000 (Normal), 103833 (RBoT) |

---

### Proposed SHAP-based knowledge distillation approach

In this step, the proposed methodology leverages SHAP (Shapley Additive Explanations) to distill knowledge from each client's local model by calculating feature importance values, which are then aggregated to form a global feature importance profile. This approach aligns each client's training process with globally relevant features, ensuring consistency across non-IID client datasets without directly sharing sensitive data.

Figure 4 illustrates the architecture of the proposed federated learning approach for IoT botnet detection using SHAP-based knowledge distillation. In this setup, multiple clients, each with non-IID data and varying numbers of classes, calculate local SHAP feature importances, denoted by \(\Phi_{k}\), for their datasets. Each client trains a local model \(f_{k}\) on its aligned data \(X_{k}^{aligned}\) and generates a normalized feature importance vector \(\phi_{k}\) based on SHAP values. The local models and SHAP-based feature importances are then uploaded to a central server. The server aggregates these local feature importances using a weighted averaging mechanism, resulting in a global feature importance profile, \(\Phi_{global}\), which reflects the contribution of each client based on its dataset size and importance. This global profile is subsequently distributed back to the clients, guiding them in refining their models based on the top global features. The approach maintains data privacy, as only feature importances and model updates are shared, not the raw data, thereby enhancing both security and robustness across heterogeneous IoT environments.

The proposed SHAP-Based Knowledge Distillation approach is shown in **Algorithm 1**. This approach ensures privacy by keeping data decentralized while achieving consistent feature importance alignment across clients. The algorithm proceeds through several key stages, beginning with feature alignment and local model training, then aggregating feature importances globally, and concluding with a training phase on selected global features and final evaluation on each client.

In the initial phase, each client \(k\) aligns its dataset \(X_{k}\) with a global feature set, ensuring that all clients operate within a unified feature space. The missing features for any client are filled with zeros, allowing each client to maintain compatibility with the global feature set while preserving privacy. Each client then trains a local model, \(f_{k}\), on this aligned dataset \(X_{k}^{aligned}\) with its corresponding labels \(y_{k}\). This local model \(f_{k}\) learns patterns specific to client \(k\)'s data distribution, representing each client's unique data characteristics.

Following local model training, each client computes SHAP values to quantify the feature importance for individual samples within its dataset. These SHAP values are aggregated across all samples to form a client-specific feature importance vector, \(\Phi_{k}\), representing the overall significance of each feature based on that client's data. This client-level importance vector \(\Phi_{k}\) captures local feature relevance, providing an essential input to the next stage of global feature aggregation.

Once each client has computed its local feature importance vector \(\Phi_{k}\), the server aggregates these vectors to create a global feature importance profile, \(\Phi_{(global)}\). This aggregation process is weighted by each client's dataset size, allowing clients with larger datasets to contribute more significantly to the global importance profile. The result is a globally aligned feature importance vector that encapsulates the relative importance of features across all clients while accounting for differences in data size.

Using this global feature importance profile, the server then selects the top \(n\) features that are most relevant across all clients. These selected features, \(X_{top}\), form a refined feature set that focuses on the key characteristics shared across all clients, thereby reducing dimensionality and enhancing training efficiency. Each client proceeds to train a new local model \(f_{k}\) using only these globally selected top features, \(X_{k}^{top}\), with corresponding labels \(y_{k}\). This ensures that each local model is consistent with the globally identified feature space while still being tailored to the local data distribution.

Finally, the trained models are evaluated on each client's original data using the selected features \(X_{k}^{top}\). The trained model \(f_{k}\) generates predictions \(\hat{y}_{k}\), which are compared with the true labels \(y_{k}\) to compute key performance metrics, such as accuracy, precision, recall, and F1-score. This final evaluation step confirms the model's effectiveness in each client's specific context, demonstrating the success of the federated learning approach in achieving robust local models aligned to a globally optimized feature set without centralizing client data.

**Algorithm 2** describes the process for training and evaluating a global model on each client in a federated learning setup using a SHAP-based knowledge distillation approach with Random Forest classifiers. The algorithm begins by selecting the top \(n\) features based on the global feature importance profile \(\Phi^{global}\), which is shared with each client. For each client \(k\), the local dataset \(X_{k}\) is aligned to include only the selected top features, creating \(X_{k}^{(top)}\). Each client then initializes a Random Forest classifier \(f_{k}^{(global)}\), which consists of multiple decision trees. In the training phase, each tree in the Random Forest is trained on a bootstrapped subset of \(X_{k}^{(top)}\) and the corresponding labels \(y_{k}\). This enables each client to learn from its local data while leveraging the globally determined important features.

After training, each client performs classification on its local dataset using the trained Random Forest model. Since each client may have a different number of classes, classification for each sample \(x_i\) in \(X_k^{(top)}\) is achieved through a majority voting mechanism across the decision trees. Each tree casts a vote for a class, and the class with the maximum votes is assigned as the predicted label \(\hat{y}_{k,i}\). Finally, the algorithm calculates evaluation metrics—accuracy, precision, recall, and F1-score—on each client's predictions to assess model performance.

By training on the selected globally relevant features, each client's model is tuned to capture the most important patterns for botnet detection, regardless of the non-IID nature of the datasets. This training process enables each client to achieve improved performance on local data while maintaining alignment with the global feature importance profile.

In this way, the global model training process enhances the overall accuracy and generalization capability of the federated learning framework across heterogeneous IoT environments, supporting robust detection of diverse botnet types.

---

## Evaluation and performance metrics

After training the local models based on the globally aligned feature set, each client evaluates its model's performance to assess its effectiveness in classifying botnet types within the distributed federated learning environment. This evaluation includes both individual client-based assessment and cross-setting comparisons to verify robustness across varying client distributions and data configurations.

### Local model evaluation on each client

Each client uses its trained model to predict botnet classifications within its local dataset. The performance metrics calculated on each client provide insights into the effectiveness of the federated learning approach at an individual level.

**Prediction using local model:** Each client uses its respective trained model \(f_{k}^{(global)}\) to classify samples in its dataset \(D_k\). Given the feature set \(X_{k}^{(top)}\), the model predicts labels \(\hat{y}_{k}\) for each data point:

\[\hat{y}_{k} = f_{k}^{(global)}(X_{k}^{(top)}) \quad (13)\]

where \(X_{k}^{(top)}\) represents the selected globally relevant features for client \(k\). These predictions form the basis for calculating the evaluation metrics.

**Calculate evaluation metrics:** The following metrics are computed to evaluate the classification performance on each client's dataset. These metrics ensure a comprehensive assessment of model accuracy and class-specific performance.

- **Accuracy:** The accuracy score reflects the overall correctness of predictions across all classes. It is calculated as:

\[Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \quad (14)\]

- **Precision:** Precision measures the proportion of true positive predictions among all positive predictions, assessing the model's ability to avoid false positives. For each class \(c\), precision is calculated as:

\[Precision = \frac{True Positives (TP)}{True Positives (TP) + False Positives (FP)} \quad (15)\]

- **Recall:** Recall calculates the proportion of actual positives that are correctly identified by the model. For each class \(c\), recall is given by:

\[Recall = \frac{True Positives (TP)}{True Positives (TP) + False Negatives (FN)} \quad (16)\]

- **F1 Score:** The F1 score is the harmonic mean of precision and recall, providing a balanced metric when classes are imbalanced. It is calculated as:

\[F1-Score = 2 \times \frac{Precision \times Recall}{Precision + Recall} \quad (17)\]

- **Confusion Matrix:** A confusion matrix provides a detailed breakdown of classification performance by showing the count of true positives, false positives, true negatives, and false negatives for each class. For each class pair \((i,j)\), the matrix entry \(CM_{i,j}\) represents the number of instances of class \(i\) that were predicted as class \(j\).

- **Classification Report:** The classification report summarizes precision, recall, and F1-score for each class, offering a detailed view of the model's performance across all classes. This report helps identify any class-specific biases or weaknesses in the model.

### Cross-Setting evaluation

To ensure robustness across different client configurations and botnet type distributions, the model is evaluated on all defined settings (Settings 1 to 3). This step assesses the federated learning model's generalization capabilities across varying data distributions.

- **Setting-Based Evaluation:** For each setting, the model performance on each client's local dataset is aggregated, and the metrics are compared across settings to evaluate the consistency of the federated approach.
- **Comparison of Metrics across Settings:** Metrics such as accuracy, precision, recall, and F1-score are averaged across clients in each setting and then compared to identify variations in performance. This comparison helps determine whether the model is robust to different configurations of botnet types and distributions across clients.

Through this comprehensive evaluation process, the federated learning model is tested for its classification accuracy and consistency across multiple IoT botnet detection environments. This cross-setting validation ensures that the model not only performs well on individual clients but also maintains robustness when data distribution and client configurations vary.

---

## Experimental results and analysis

In this section, the experimental results and analysis are presented through relevant figures and tables to provide a clear understanding of the proposed model's performance across different federated learning settings. Comparisons are drawn across various settings to demonstrate the model's robustness in handling data heterogeneity, client diversity, and non-IID distributions, which are inherent challenges in IoT botnet detection.

### Experimental setup

The experimental setup for this research was conducted on Google Colab, utilizing Python libraries to implement and test our federated learning framework for IoT botnet detection. Key libraries included NumPy and Pandas for data handling and scikit-learn for machine learning models and preprocessing tasks, such as RandomForestClassifier for model training, StandardScaler for feature normalization, and LabelEncoder for encoding categorical data. SHAP was essential in calculating feature importances, enabling privacy-preserving aggregation of knowledge across clients. To avoid potential data leakage and ensure a fair evaluation, we employed a 70% training and 30% testing split in all experiments. Importantly, within the training portion, we further separated a validation set (typically 20% of the training data) to optimize hyperparameters such as the number of trees, maximum depth, and minimum samples per split in the Random Forest classifier. This means the final test set remained entirely unseen during hyperparameter tuning, effectively preventing test data leakage and overfitting. Only after selecting the best-performing hyperparameters on the validation set was the model evaluated on the test set.

### Classification results

Before describing the results of federated settings, we first present the classification results using a centralized approach with Random Forest on 30% of the test dataset. **Table 4** provides the accuracy, precision, recall, and F1-score for the models trained on three different datasets. These results demonstrate the performance of the centralized model using optimized hyperparameters across each dataset, achieving near-perfect classification scores across all metrics, with accuracy values ranging from 99.95 to 99.99%.

**Table 4: Classification results of centralized approach with best hyperparameters for different datasets**

| Dataset | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|---------|--------------|---------------|------------|---------------|
| MedBIoT | 99.96 | 99.96 | 99.97 | 99.97 |
| N-BaIoT | 99.99 | 99.99 | 99.99 | 99.99 |
| NCC2 | 99.95 | 99.95 | 99.95 | 99.95 |

**Table 5** presents the classification performance metrics for each client in Federated Setting 1. Each client's performance is evaluated in terms of precision, recall, and F1-score for their respective botnet classes, demonstrating the model's effectiveness across different botnet types. Client 1, with classes Bashlite (0) and Normal (2), achieved an overall accuracy of 99.94% with both classes showing high precision and recall. Similarly, Client 2, comprising Mirai (1) and Normal (2), achieved an accuracy of 99.97% with minimal variance between precision, recall, and F1-score, suggesting balanced classification. Client 3, containing Normal (2) and Torii (3) classes, achieved an accuracy of 99.99% showing the model's robustness in distinguishing between these classes. Overall, the results highlight the framework's high classification accuracy and balanced performance across heterogeneous client datasets.

**Table 5: Classification performance metrics for each client in federated Setting 1**

| Client | Class | Precision | Recall | F1-score |
|--------|-------|-----------|--------|-----------|
| Client 1 | 0 (Bashlite) | 0.99866 | 0.99965 | 0.99915 |
| | 2 (Normal) | 0.99982 | 0.99931 | 0.99956 |
| | Accuracy | 0.99942 | | |
| | Macro avg | 0.99924 | 0.99948 | 0.99936 |
| | Weighted avg | 0.99942 | 0.99942 | 0.99942 |
| Client 2 | 1 (Mirai) | 0.99959 | 0.99993 | 0.99976 |
| | 2 (Normal) | 0.99994 | 0.99967 | 0.99980 |
| | Accuracy | 0.99978 | | |
| | Macro avg | 0.99977 | 0.99980 | 0.99978 |
| | Weighted avg | 0.99978 | 0.99978 | 0.99978 |
| Client 3 | 2 (Normal) | 1.00000 | 0.99985 | 0.99992 |
| | 3 (Torii) | 0.99983 | 1.00000 | 0.99991 |
| | Accuracy | 0.99992 | | |
| | Macro avg | 0.99991 | 0.99992 | 0.99992 |
| | Weighted avg | 0.99992 | 0.99992 | 0.99992 |

The confusion matrices shown in **Fig. 5** illustrate the classification performance of the proposed federated learning model on each client dataset in Setting 1. The low number of misclassifications across all clients reinforces the effectiveness of the proposed approach in handling heterogeneous botnet types in non-IID datasets, showcasing its capability to accurately detect IoT botnet intrusions across different devices and environments in a federated setting.

**Table 6** presents the classification performance of the local model across multiple clients under Setting 2, capturing various botnet types such as Bashlite, Mirai, and Torii along with normal traffic. Each client dataset reflects a different distribution of botnet and normal traffic, allowing us to evaluate the model's adaptability and robustness in handling diverse IoT botnet attacks. The high performance metrics for all clients underscore the effectiveness of the proposed federated learning approach, which leverages SHAP-based feature selection to enhance model interpretability and accuracy while preserving data privacy across distributed IoT devices. The near-perfect classification results highlight the model's robustness in identifying botnet attacks, even in non-IID data settings, where client datasets vary significantly in botnet type and normal traffic distribution.

**Table 6: Classification report for each client in Setting 2**

| Client | Class | Precision | Recall | F1-Score | Accuracy |
|--------|-------|-----------|--------|-----------|----------|
| Client 1 | Bashlite | 0.99921 | 0.99952 | 0.99936 | 0.99957 |
| | Normal | 0.99976 | 0.99959 | 0.99968 | |
| | Overall | 0.99957 | 0.99957 | 0.99957 | |
| Client 2 | Mirai | 0.99973 | 0.99987 | 0.99980 | 0.99982 |
| | Normal | 0.99989 | 0.99978 | 0.99984 | |
| | Overall | 0.99982 | 0.99982 | 0.99982 | |
| Client 3 | Normal | 0.99997 | 0.99992 | 0.99995 | 0.99994 |
| | Torii | 0.99991 | 0.99997 | 0.99994 | |
| | Overall | 0.99994 | 0.99994 | 0.99994 | |
| Client 4 | Bashlite | 0.98501 | 0.99169 | 0.98834 | 0.99474 |
| | Mirai | 0.99489 | 0.99083 | 0.99286 | |
| | Normal | 0.99962 | 0.99943 | 0.99953 | |
| | Overall | 0.99474 | 0.99474 | 0.99474 | |
| Client 5 | Bashlite | 0.99921 | 0.99974 | 0.99947 | 0.99976 |
| | Normal | 0.99986 | 0.99959 | 0.99973 | |
| | Torii | 0.99997 | 0.99997 | 0.99997 | |
| | Overall | 0.99976 | 0.99976 | 0.99976 | |
| Client 6 | Mirai | 0.99912 | 0.99970 | 0.99941 | 0.99962 |
| | Normal | 0.99981 | 0.99978 | 0.99980 | |
| | Torii | 0.99984 | 0.99935 | 0.99960 | |
| | Overall | 0.99962 | 0.99962 | 0.99962 | |
| Client 7 | Bashlite | 0.98372 | 0.99211 | 0.98790 | 0.99582 |
| | Mirai | 0.99468 | 0.98985 | 0.99226 | |
| | Normal | 0.99957 | 0.99938 | 0.99947 | |
| | Torii | 0.99972 | 0.99941 | 0.99956 | |
| | Overall | 0.99582 | 0.99582 | 0.99582 | |

The confusion matrices shown in **Fig. 6** for each client illustrate the performance of the local model in distinguishing between botnet classes (Bashlite, Mirai, Torii, etc.) and normal traffic across the various data partitions. Here 67.67% of data was used for training and 33.33% was used for testing. Each matrix shows a high concentration of correct classifications along the diagonal, indicating accurate predictions for each botnet type and normal traffic. The minimal off-diagonal entries, which represent misclassifications, affirm the model's robustness and effectiveness in identifying botnet traffic types with high accuracy across different client distributions. This strong performance demonstrates the capability of the federated learning framework to generalize well across heterogeneous IoT environments, achieving reliable detection across varied and non-IID client data.

**Table 7** summarizes the classification performance of the federated learning model in Setting 3, where each client has a unique mix of botnet and normal data from different datasets. Across all clients, the model achieved nearly perfect scores in accuracy, precision, recall, and F1, demonstrating its strong capability to accurately identify and differentiate between normal traffic and various botnet types such as Bashlite, Mirai, Torii, Neris, and RBoT. This consistency across diverse data sources and botnet types highlights the model's adaptability and effectiveness in a federated, non-IID environment, underscoring its potential for robust IoT security.

**Table 7: Classification results for Setting 3 with non-IID data distribution from different datasets**

| Client | Botnet Class | Precision | Recall | F1-Score | Accuracy |
|--------|--------------|-----------|--------|-----------|----------|
| Client 1 | Bashlite | 0.99999 | 0.99992 | 0.99996 | 0.99992 |
| | Normal | 0.99942 | 0.99942 | 0.99967 | |
| | Weighted avg | 0.99992 | 0.99992 | 0.99992 | |
| Client 2 | Mirai | 0.99998 | 1.00000 | 1.00000 | 1.00000 |
| | Normal | 1.00000 | 1.00000 | - | |
| | Weighted avg | 1.00000 | 1.00000 | 1.00000 | |
| Client 3 | Normal | 0.99888 | 0.99908 | 0.99898 | 0.99979 |
| | Torii | 0.99894 | 0.99872 | 0.99883 | |
| | Weighted avg | 0.99979 | 0.99979 | 0.99979 | |
| Client 4 | Neris | 1.00000 | 1.00000 | - | 1.00000 |
| | Normal | 1.00000 | 1.00000 | 1.00000 | |
| | Weighted avg | 1.00000 | 1.00000 | 1.00000 | |
| Client 5 | Normal | 0.99998 | 0.99997 | 0.99998 | 0.99999 |
| | RBoT | 0.99999 | 1.00000 | 0.99999 | |
| | Weighted avg | 0.99999 | 0.99999 | 0.99999 | |

The confusion matrices of binary classification for different clients in **Figs. 7 and 8** for Setting 3 illustrate the performance of the federated model across different clients, each containing unique botnet types and normal traffic from separate datasets. In all cases, the model demonstrates exceptional accuracy, with nearly all samples classified correctly for each client. For instance, Client 1 achieved near-perfect classification for Bashlite and normal traffic, while Client 2 correctly classified Mirai and normal data with minimal errors. Similarly, Clients 3, 4, and 5 show high accuracy in distinguishing between various botnet types and normal traffic, as reflected by the minimal misclassifications in each matrix.

---

### Discussion on the effectiveness of the proposed approach and answer to the research questions

The results from Settings 1 to 3 demonstrate the robustness and adaptability of our proposed federated learning approach in handling heterogeneous, non-IID data distributions across multiple IoT clients. In Setting 1, high accuracy and balanced performance metrics were achieved across all clients, which had varied botnet types and normal traffic. This illustrates the model's effectiveness in detecting attacks in a distributed environment without centralized data sharing. Setting 2 further validated this by expanding client diversity and maintaining excellent performance, showcasing the approach's adaptability in scenarios with even more varied botnet and normal data distributions. Setting 3 involved data from multiple datasets and botnets (Bashlite, Mirai, Torii, Neris, RBoT), demonstrating the model's capability to generalize across different botnet families.

The computational complexity of the proposed federated learning approach for IoT botnet attack detection is optimized to balance efficiency, scalability, and the constraints of IoT environments. Local model training, employing Random Forest classifiers, scales with \(O(t \cdot n \cdot \log(n))\) per tree, where \(t\) is the number of trees and \(n\) is the number of samples per client. This ensures efficient training on non-IID client datasets while distributing computational demands across devices. The SHAP-based feature importance computation, essential for explainability and knowledge distillation, operates with a complexity of \(O(T \cdot L \cdot D)\), where \(T\) is the number of trees, \(L\) is the maximum depth of a tree, and \(D\) is the number of features. The hyperparameter tuning selects T (Number of Trees) as 10, the maximum depth L of each tree is set to None (default in Random Forest from Scikit-learn), resulting in variable tree depth, and D (Number of Features) varies across datasets as shown in Fig. 2. Although SHAP computations require considerable processing power, performing them locally on the devices ensures that the central server remains efficient and unaffected by additional computational load. Global feature aggregation, which performs weighted averaging of SHAP values from clients, is computationally lightweight with a complexity of \(O(k \cdot D)\), where \(k\) is the number of clients. The overall complexity of the framework can thus be approximated as \(O(k \cdot (t \cdot n \cdot \log(n) + T \cdot L \cdot D))\), dominated by the local computations across \(k\) clients. By leveraging SHAP to prioritize important features, the approach effectively reduces data dimensionality, further optimizing computational and communication overhead. This design achieves high detection performance while ensuring scalability and feasibility for IoT security, even under the constraints of heterogeneous and resource-limited devices.

The proposed approach for IoT security is practically implemented by deploying local intrusion detection systems on IoT devices or gateways, where models are trained on device-specific data to ensure privacy. Using SHAP, feature importance is computed locally, enabling explainability and prioritization of critical features. Aggregated insights are securely shared with a central server for global feature optimization, facilitating coordinated detection across devices. This distributed and privacy-preserving design ensures scalability, adaptability to evolving threats, and feasibility for resource-constrained IoT environments.

Our approach addresses limitations in centralized ML/DL methods, which often struggle with privacy concerns and the inability to handle non-IID data effectively. By utilizing federated learning, our method eliminates the need for direct data sharing, preserving client privacy and meeting data security requirements crucial for IoT environments. Moreover, centralized approaches often fail to adapt to the diverse, evolving nature of IoT botnets due to their static nature. In contrast, our approach employs SHAP-based feature importance aggregation, enabling dynamic updates that account for varied client-specific patterns, effectively tackling the evolving attack architectures. Furthermore, traditional FL models lack interpretability, which is essential for identifying attack patterns in IoT networks. Our use of SHAP-based knowledge distillation not only enhances the interpretability but also ensures that only essential information is shared with the central server. This targeted knowledge sharing and aggregation process reduces communication overhead and enhances the model's performance across distributed clients, overcoming common limitations found in traditional FL. Overall, our proposed federated framework provides a privacy-preserving, interpretable, and highly effective solution for IoT botnet detection, addressing critical gaps in both centralized ML/DL and standard FL approaches.

The research questions listed in the introduction section are addressed based on our findings, and the corresponding answers are provided below.

- **RA1:** Federated learning is effectively utilized in our framework by allowing decentralized model training directly on each IoT client, preserving privacy by ensuring that raw data is never shared. Through this approach, the framework handles the heterogeneity and non-IID nature of IoT data by leveraging SHAP-based feature alignment, which enables the aggregation of feature importance scores from diverse devices. This ensures robust detection of botnet intrusions across heterogeneous IoT environments while addressing privacy and scalability concerns.
- **RA2:** The integration of SHAP-based knowledge distillation enhances model interpretability by quantifying feature importance locally at each client, sharing only distilled insights with the central server. This alignment of globally relevant features ensures consistency across diverse IoT devices and provides transparency into the decision-making process of the global model. This interpretability not only improves trust in the framework but also enhances detection accuracy by focusing on the most critical features.
- **RA3:** Decentralized feature aggregation plays a crucial role in improving scalability and adaptability by allowing each client to train on locally relevant data and contributing feature importance scores to the global model. This approach enables the framework to detect botnet patterns effectively across diverse and resource-constrained IoT devices. The decentralized design reduces communication overhead, supports large-scale deployment, and allows the system to adapt to varying data distributions.
- **RA4:** The proposed framework addresses the challenges posed by evolving botnet architectures and resource constraints by incorporating continuous learning through decentralized training and SHAP-based feature prioritization. By focusing on essential feature insights, the framework adapts to changing botnet behaviors without requiring extensive computational resources on IoT devices. This flexibility ensures effective detection of new and emerging threats while maintaining efficiency in resource-limited environments.

---

### Comparison with existing approaches

**Table 8** provides a comparative analysis of our proposed decentralized approach with various existing centralized and federated learning models for IoT botnet intrusion detection. Key dimensions considered include the ability to handle data heterogeneity, dimensionality reduction, non-IID data handling, explainability, and model performance metrics like accuracy, precision, and recall.

Unlike traditional centralized methods like DNN, SGC, and CNN-LSTM, which lack the capability to handle heterogeneous data and non-IID distributions effectively, our proposed approach leverages a decentralized FL setup. This setup enables clients to locally process diverse data distributions while safeguarding privacy, thus addressing the limitations of centralized models that rely on uniform data processing. Furthermore, unlike most centralized models that ignore dimensionality reduction, our approach incorporates SHAP-based feature selection, which enhances interpretability and reduces computational complexity.

Comparing with existing FL approaches, such as FDL, our model exhibits superior performance in terms of accuracy, precision, and recall, reaching nearly 99.99% across these metrics. This result highlights the robustness of our model in accurately detecting botnet attacks while addressing both data heterogeneity and non-IID issues, which are often neglected in other approaches. Moreover, our use of SHAP-based explainability sets us apart by providing model interpretability, which is generally absent in the existing methods.

**Table 8: Comparison of the proposed approach with diverse existing approaches**

| Model with reference | Dataset | Training approach | Accuracy (%) | Precision (%) | Recall (%) |
|----------------------|---------|-------------------|---------------|---------------|-------------|
| DNN [31] | N-BaIoT | Centralized | 97.21 | 91.41 | 87.31 |
| A³E [32] | N-BaIoT | Centralized | 98.00 | 99.00 | – |
| SGDC [33] | N-BaIoT | Centralized | 98.43 | – | 98.42 |
| IHHO-NN [34] | N-BaIoT | Centralized | 98.07 | 97.04 | 98.73 |
| ER-VEC [35] | N-BaIoT | Centralized | 95.64 | – | – |
| WCC [38] | N-BaIoT | Centralized | 96.70 | 94.90 | 94.70 |
| GWO [39] | N-BaIoT | Centralized | 98.97 | – | – |
| LGBA-NN [40] | N-BaIoT | Centralized | 90.00 | 85.23 | 90.00 |
| CNN-LSTM [36] | N-BaIoT | Centralized | – | 94.00 | 89.00 |
| RNN [41] | N-BaIoT | Centralized | 89.75 | – | – |
| FDL [25] | N-BaIoT | FL | 98.37 | 83.31 | 86.32 |
| RF with Chi-Squared [42] | MedBIoT | Centralized | 95.30 | 95.00 | 98.00 |
| RF [43] | MedBIoT | Centralized | 96.17 | 96.92 | 92.17 |
| DT [44] | MedBIoT | Centralized | 95.16 | 95.84 | 95.16 |
| PSO-RF [15] | MedBIoT | Centralized | 96.20 | 99.00 | 93.27 |
| One-class KNN [45] | MedBIoT | Centralized | 98.00 | – | 98.00 |
| AU-ANN [46] | MedBIoT | Centralized | 99.72 | 99.82 | 99.83 |
| DT [47] | MedBIoT | Centralized | 99.41 | 99.36 | 99.38 |
| KNN [48] | NCC2 | Centralized | 95.91 | 95.07 | 80.00 |
| B-CAT [49] | NCC2 | Centralized | 99.87 | – | 97.38 |
| Bi-LSTM [37] | NCC2 | Centralized | 99.93 | – | – |
| **Proposed** | **MedBIoT, N-BaIoT, NCC2** | **FL** | **99.99** | **99.99** | **99.99** |

*Note:* "–" indicates not mentioned in the respective reference.

---

## Conclusion

In this research, we proposed a novel privacy-preserving federated learning framework for IoT botnet detection, employing SHAP-based knowledge distillation and decentralized feature aggregation to address the inherent challenges of data heterogeneity and non-IID data distributions. Our approach enables each client to independently train on its localized, diverse dataset, generating interpretable SHAP values for feature importance and sharing distilled insights with a central server. This aggregation of client-specific knowledge allows for a robust, adaptable global model capable of accurately detecting a range of botnet attacks, including Bashlite, Mirai, and Torii, in complex IoT environments. The experimental results demonstrate that our method achieves near-perfect accuracy, precision, and recall across multiple settings, underscoring the effectiveness of our decentralized model in maintaining high detection performance while preserving data privacy. This study not only addresses the limitations of traditional centralized models and existing FL methods in handling heterogeneous, distributed IoT data but also contributes a scalable, interpretable solution for IoT security. Future research could explore the application of our approach to broader IoT environments with an even wider array of devices, enhance the system's resilience to emerging botnet variants, and investigate adaptive federated mechanisms that allow continuous learning and improvement of the global model in real-time IoT networks.

