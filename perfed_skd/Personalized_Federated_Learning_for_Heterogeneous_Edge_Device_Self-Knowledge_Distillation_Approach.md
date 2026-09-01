# Personalized Federated Learning for Heterogeneous Edge Device: Self-Knowledge Distillation Approach

**Neha Singh**, Graduate Student Member, IEEE, **Jatin Rupchandani**, and **Mainak Adhikari**, Senior Member, IEEE

---

## Abstract

Federated learning (FL) has become increasingly popular and distributes machine learning models among a large set of resource-constraint edge devices without transferring data to the centralized server. FL seeks to learn a common global model using a set of local models, trained in distributed edge devices, coordinated by a central device. However, a set of slow processing edge devices fail to transmit the locally updated model parameters to the centralized device due to a lack of bandwidth connectivity. Additionally, the data residing across edge devices is statistically diverse, i.e., the distribution of non-IID data. Such communication overhead and the variety of data distribution are two major obstacles to the practical application of FL. Motivated by the challenges, in this research work, we develop a self-knowledge distillation-enabled Personalized Federated Learning framework, namely PerFed-SKD. By enabling edge devices to transfer knowledge from older local models to more recent personalized models, the proposed PerFed-SKD speeds up the process by recalling the historical personalized knowledge for the most recent initialized model. Extensive experiments on two publicly available datasets, i.e., MNIST and EMNIST with various data distribution settings demonstrate the outperformance of the proposed PerFed-SKD over the state-of-the-art methods.

**Index Terms**—Edge computing, personalized federated learning, self-knowledge distillation, data heterogeneity.

---

## I. Introduction

Due to state-of-the-art performance and simplicity in implementation, the Deep Neural Network (DNN) is gaining popularity in the broad variety of potential applications of Machine Learning (ML) techniques. To fully realize the potential of these applications, massive amounts of data in the complicated Internet-of-Things (IoT) environment are used to train the DNN model. Most IoT data is traditionally processed on a centralized resource-rich cloud server. However, this raises serious privacy preservation concerns and requires huge communication bandwidth, which is not acceptable for real-time applications. To overcome this issue, there has been a rise in the quantity and diversity of edge devices, including edge routers, smartphones, tablets, consumer and industrial robots, smart mobile phones, etc. that process the data at the edge of the network. However, the lack of computation and storage capacity in the local edge devices makes it difficult to achieve the desired inference accuracy while training with the standard DNN model using limited data [1].

Nowadays, Federated learning (FL) opens new horizons for distributed ML due to its improvements in handling the aforementioned privacy concerns, accuracy, and resolving the issue of data silos. In FL, active clients, i.e., local edge devices jointly train an ML model without disclosing their personal information [2]. McMahan et al. have developed the first FL technique in a distributed environment, namely Federated Averaging (FedAvg). In FedAvg, the centralized cloud server aggregates the local model parameters collectively and delivers new aggregated model parameters to the local edge devices for future training cycles. Finally, a unified and enhanced global model is transmitted to edge devices.

FL is a promising distributed ML technique, where uploading data is not possible to safeguard the privacy of the edge devices. Therefore, FL is well adapted for real-world situations like analyzing time-critical data for immediate decision-making and recommendation. However, the standard FL technique suffers from various challenges, including:

**Data Heterogeneity:** The distribution of inherently non-independent and identically distributed (non-IID) data across edge devices becomes a significant issue with the prevalence of FL. Each client generates its own data, some of which may be unreliable or only include a portion of the total features and identifiers. Due to non-IID data, the global FL model produces better accuracy than local models in some of the edge devices during training. In such a scenario, some edge devices get worse model parameters from the centralized server than its previous model parameters [3].

**Resource-Constraint Edge Device:** Typically, edge computing involves resource-constraint devices that have limited storage and processing capacity. As a result, if the cloud server frequently sends and gets models to and from edge devices due to limited processing capacity, it is simple to become a communication bottleneck [4].

**Communication Overhead:** In the traditional FL process, there are several information exchanges required between the edge device and the cloud server [5]. This exchange of information induces a substantial communication load, which jeopardizes the functioning efficiency.

Many studies have been conducted to mitigate data heterogeneity through FL personalization. The goal of Personalized Federated Learning (PFL) is to enable every edge device to develop a customized model that excels at edge-specific tasks. PFL requires generalization in addition to personalization to take advantage of shared knowledge from collaboration and suit the local data distribution [6]. The main challenge with PFL is to make a proper balance between shared global knowledge and local model knowledge. In FL, at the beginning of each communication round, the local model parameters of each client are updated using the most recent global parameters. Such updation removes all historical information from local storage while reducing performance accuracy. Thus, in the next communication round, the edge devices begin to train the model from the initial phase. The criticality of this process increases during heterogeneous data distribution among edge devices. Additionally, Self-Knowledge Distillation (SKD) adaptively personalizes models for individual devices over time. As new data becomes available on a device, the model continues to distill knowledge from the global model, enabling continuous learning and adaptation to changing user preferences or data distributions.

Motivated by the above challenges and advantages of SKD, we develop a novel SKD-enabled PFL framework (PerFed-SKD). In place of conventional PFL, PerFed-SKD adopts the concept of SKD where the prior personalized model transmits historical personalized knowledge to the current local model. Therefore, PerFed-SKD remembers historical local knowledge and achieves a fair trade-off between personalization and generalization. This prior local model offers a different perspective on local data in feature space. Thus, the distillation of self-knowledge produces an implicit ensemble performance with knowledge distillation. In the next round, the model accuracy of each local model is compared with the global model at the central server. The updated global model is reverted back to those edge devices having less local accuracy compared to the global one. The global model parameters, received by the edge devices are then trained under the guidance of previous personalized models that forwarded the personalized knowledge to edge devices for local model training using SKD. The rest of the edge devices train their model using existing model parameters. Therefore, PerFed-SKD saves the knowledge in local storage and makes a balance between generalization and personalization. The accuracy comparison method on the central server for model updation reduces the communication overhead as the global model is shared with a subset of local devices that have lesser accuracy than the global model. The contributions of the research work are summarized as follows.

- We design a new PFL mechanism in edge networks for real-time applications, namely PerFed-SKD. We explore the personalized data of each edge device that had traditionally been forgotten while initializing the global aggregated model parameters.
- We develop an efficient PFL framework using SKD where a teacher model is trained on the central server to transfer the historical personalized knowledge to the student model at edge devices. This reduces the training burden on resource-constraint devices and creates a balance between generalization and personalization.
- In the aggregation round, a threshold is set to compare the accuracy of the global model and all the local models. In each training round, local models are updated based on threshold accuracy. This reduces the communication overhead and gives extra space to resource-constraint devices for better decision-making.
- Extensive experiments are conducted over two publicly available image datasets, i.e., MNIST and EMNIST using the state-of-the-art FL methods and the proposed PerFed-SKD. The simulation results demonstrate the effectiveness of the proposed PerFed-SKD method.

The remaining sections are organized as follows. Section II describes the related studies of the existing FL techniques. Section III highlights the system model and problem statement. Section IV describes the proposed PerFed-SKD mechanism. The performance and comparative analysis are described in Section V. Section VI concludes the work.

---

## II. Related Work

This section examines several challenges facing existing FL algorithms. As part of FL, we discuss three sub-topics: Data Heterogeneity that enforces personalization works, work to reduce communication overhead to improve convergence speed and work to improve global-model performance.

The main reason behind devices participating in FL is to learn a global model that works better than a local one. Existing studies, such as [7] and [8], have shown that during data heterogeneity (when non-IID data is spread across devices), the locally trained models work better than the model learned globally, which ignores the main incentive of FL [6]. So, several researchers have looked into the idea of "personalizing" FL models to make them work better with data from non-IID users. For example, Fallah et al. [9] came up with the PerFedAvg method, where a first-order adaptation term to the client loss functions allows it to be adapted in a single step to the client dataset. Jiang et al. [10] developed a three-stage training algorithm to improve personalization. Hanzely and Richtarick added a learnable measure so that clients could control how many local and global models were mixed together [7]. Qu et al. observed that FL-based solutions suffer from data heterogeneity and privacy disclosure and proposed the PFL with the generative adversarial network to store the previous model information and enhance the prediction accuracy [11]. Jiang et al. proposed the PFL framework for dynamic sparse training to maintain the performance of the local model by reducing communication overhead in a cloud-edge environment that solves the data heterogeneous and solves non-IID data handling using FL [12]. Zhang and Xu presented the pre-clustering-based PFL for generating personalized models locally while reducing overall loss during training [13].

Even though numerous efforts have been made by authors, many other factors affect how PFL is deployed, i.e., costly communication overhead between the central server and edge devices. In the standard FL, the latest global model performs better on global distribution rather than the different local distribution, which produces biased results [14]. The innovative approach, LG-FedAvg [15], seeks to simultaneously enhance model personalization and communication effectiveness. Many authors used L2-norm and other regularizers to help the edge devices recall historical personalized knowledge. Knowledge distillation for neural networks was first presented by [16] and has demonstrated exceptional effectiveness in various fields. According to a few recent papers [17], [18], a teacher and student model with the same structure could increase the generalization ability of students relative to the teacher. The idea of SKD is also used in a recent preprint FedLSD [18] to improve the global model in a non-IID setting.

In summary, the literature indicates that the use of FL and PFL concepts in edge networks is a growing field with many successful applications in real-time applications. Furthermore, the use of SKD in PFL provides a promising solution for training the model with personalized data. The comparative analysis of the proposed PerFed-SKD framework and existing works for real-time data analytics over five key attributes is depicted in Table I.

**Table I: Comparative Investigation of the Study Gap in Literature**

| Existing Work | Deep Neural Networks | PFL | Data Heterogeneity | Communication Overhead | Resource Constraint Edge Devices |
|---------------|---------------------|-----|-------------------|----------------------|--------------------------------|
| [6] | × | × | × | × | × |
| [9] | × | × | × | × | × |
| [12] | × | × | × | × | × |
| [13] | × | × | × | × | × |
| [15] | × | × | × | × | × |
| [16] | × | × | × | × | × |
| [18] | × | × | × | × | × |
| **Our Work** | **✓** | **✓** | **✓** | **✓** | **✓** |

The majority of past work relied on public or proxy data, which required careful analysis and even advanced knowledge of clients' private data. Motivated by the above challenges, we use SKD in the PFL to train the large neural network on a cloud server, known as the teacher model that transfers the knowledge to edge devices, known as a student model. We also take care of the communication overhead and resource constraint nature of devices by applying the device selection method on the cloud server that gives space to edge devices and reduces the frequent model exchange to all edge devices. Without adding any new knowledge or data, the proposed method maintains the traditional FL system's assumption. Additionally, the proposed method fully utilizes available resources and is easily adaptable to popular PFL systems.

---

## III. System Model and Problem Formulation

In this section, we first give a high-level view of our proposed framework followed by the problem statement with an empirical observation about the PFL problem.

### A. System Model

**Fig. 1. Proposed PerFed-SKD framework in edge networks.**

The proposed PerFed-SKD is a customized FL architecture that enhances both communication and inference efficiency. Like traditional FL frameworks, PerFed-SKD allows each participating device to train using a personalized model that can be built through SKD. It contains multiple edge devices along with a cloud server. The cloud server consists of the following elements:

**The Coordinator** oversees the entire FL process and works as a top-level supervisor in the cloud. It gives directives to other server components, regulates the group of edge devices, and synchronizes FL responsibilities amongst them. The coordinator must have the information on all the devices and also keep track of the activities being performed.

**The Teacher Model** is responsible for training the dense network and transferring their knowledge to the student model through SKD. Initially, the global model is trained by the central server using the predefined dataset. Next, the newly developed student model is distributed to the local edge devices.

**The Aggregator** receives all the local model parameters from the edge devices to generate a global model for training in the next iterations. Thus, local model parameters are the input, and global model parameters are the output of an aggregator.

**The Device Selector** selects a subset of edge devices to participate in each training round. The selection is based on the global model accuracy comparison in which all local models' accuracy is compared with global aggregated model accuracy. Particular edge devices that have lower accuracy models are selected for the communication rounds.

**The Communication Manager** is responsible for handling the interaction between the cloud server and edge device. Various types of exchange in FL process like sending task request messages to the edge device, transferring local updated model parameters to server and teacher model to student model with other auxiliary exchange of data.

The following are the components of the edge device:

**The Controller** works as a manager in the local edge devices (i.e., on the client side) that handles the execution of local processes and produces instructions for the neighboring edge device.

**The System Monitor** collects all system-level information of the local edge devices including current load, battery backup, etc. Based on this information and statistics, the controller determines to approve the server's request for work.

**The Student Model Manager** is responsible for preparing the local model. Initially, the global model parameters of the centralized server are initialized to the local model for training purposes. After training the local model, the model parameters are saved in the local storage as personalized model parameters of the edge devices and recall the model parameters in the future using knowledge transfer. To develop the abovementioned logic, we introduce the SKD strategy by initializing the local model from the previous personalized model.

**Data Manager** is responsible to gather and handle the dataset in real-time scenarios for model training and testing.

**Training Optimizer** is responsible for training the distributed edge devices with local model parameters. A proximal term is incorporated as an objective function for local training that is created to train the standard FL methods.

### B. Problem Formulation

The main target of the proposed work is to train personalized models using the concept of FL in a collaborative edge-cloud network. Let us consider that a total of M edge devices are connected to a centralized device. Each edge device m ∈ [M] consists of its private dataset D_m for local training. The data sample has a pair, denoted as (k,l), where k and l represent the input and the label, i.e., l ∈ [1,C], respectively. The complete dataset is represented as D = ∪_{m=1}^{M} D_m. The main objective of the standard FL method is to reduce the loss rate by finding an optimal global model ω over the dataset D_m.

$$\text{minimize} \qquad F(\omega) = \sum_{m=1}^{M} \frac{|D_m|}{|D|} F_m(\omega) \quad (1)$$

where F_m(ω) represents loss function of the m-th edge device. Here, F(ω) is the global objective function of the weighted average of local objective functions (F_m(ω)). The standard FedAvg iterates between local edge devices and centralized server for local training and global aggregation, respectively. In each training phase, the centralized server sends the global model ω to each participating edge device m ∈ [M], which is used to initialize its local model ω_m. The new global model is generated by aggregating the local model parameters of the selected edge devices that have different data distributions. With the lowest possible loss rate, the global model outperforms the local ones in this case. This is a representation of the difference in test accuracy for personalized tests between local models on edge devices and the global model in the centralized server. Similarly, the selection of edge devices in the communication round is also a big challenge. During the training phase, if the local model parameters of some edge devices are not considered, then identifying the global model training starting point by recalling the previous personalized knowledge is a challenging task. Besides that, if all the edge devices are selected in each round then the problem of resource constraint arises. So, apart from Problem 1, we have to effectively select a subset of edge devices, which is explained in Section IV.

---

## IV. PerFed-SKD: Self-Knowledge Distillation Aided Personalized Federated Learning

**Fig. 2. Workflow of proposed PerFed-SKD framework.**

To solve problem 1, we propose a new SKD-aided PFL (PerFed-SKD) method. The workflow of the proposed PerFed-SKD model is depicted in Fig. 2. The whole process consists of seven phases: a) The centralized server broadcasts the initial global model ω^t to all the edge devices S in the network; b) All the edge devices S ∈ |M| start to train the model using SKD by setting the local model parameters ω_m^t at t-th iteration based on the received global parameter; c) After training, the distributed edge devices store the trained local model ω_m^{t+1} as the teacher model V_m for next training phase; d) Next, the local model ω_m^{t+1} is sent back to the centralized server for further aggregation; e) The central server aggregates all the received model parameters to develop a new global model. f) Then, the device selector in the central server selects the local edge devices whose accuracy is below the threshold value. g) Finally, the selected devices are then sent the newly aggregated model for better performance in the next iteration.

To maintain the historical personalized knowledge, we continue to use the personalized model V_m for local training supervision m-th edge device in the remaining communication rounds. To store the personalized model for further training, we update the local model (ω_m^t) parameters using personalized model V_m. We adopt the concept of SKD policy to transfer personalized knowledge from past local models to current ones. For executing this process, we select a set of edge devices for transferring local model parameters to the centralized server in each iteration. This process reduces the overburden of edge devices and makes the federated process fast for resource-constraint devices. Further, the SKD policy is adopted on the edge device local updates phase to generate and store the most recent personalized model V_m. The personalized knowledge of each edge device helps to handle the data heterogeneity among edge devices. To do this, the empirical loss of edge device F_m(ω_m^t) is combined with the loss function φ_m(ω_m^T) of the m-th edge device, and the distillation loss is formulated as follows.

$$\phi_m(\omega_m^t) = f_m(\omega_m^t) + \lambda Lx(V_m)||x(\omega_m^t) \quad (2)$$

Here, λ represents the hyperparameter for controlling the process of SKD. f_m(ω_m^t) represents the cross entropy loss of m-th edge device. L denotes the divergence function between current local prediction x(ω_m^t) and past personalized prediction x(V_m). In the place of F_m(ω_m^t), the m-th edge device changes the local weights following the local objective φ_m(ω_m^t). As a result, Stochastic Gradient Descent is used to update the local weights ω_m^t, defined as shown below.

$$\omega_m^t = \omega_m^t - \eta \delta \phi_m(\omega_m^t, V_m) \quad (3)$$

where η represents the learning rate.

The detailed methodology of the proposed PerFed-SKD model is summarized as follows. The logic of global model training in the centralized server is depicted in Algorithm 1. Initially, the centralized server initializes the global model using a predefined dataset D (Line 2). During global model training, the proposed PerFed-SKD runs T rounds (Line 3). In each round, the centralized server randomly selects a subset S of edge devices for updating the local model using the global model, and the remaining edge devices update the local model with the existing model parameters. The subset of edge devices is selected based on global model accuracy (Line 4). If the local model's accuracy is less than the global model's accuracy then add that local model holder edge device in a subset S (Line 5-6). Next, the centralized server transmits the global model parameters to the set of selected edge devices belonging to a subset S in the network (Line 7-8). The selected edge devices perform the local training in the next iteration using the updated global model whereas the remaining edge devices train the model again with the existing model parameters (Line 9). Finally, the central server receives all the local model parameters from the selected edge devices (Line 10) and aggregates them for developing an updated global model for the next iteration (Line 11).

### Algorithm 1: PerFed-SKD: Global Model Training

**Input:** Edge devices: m, Communication round: t, Accuracy Threshold: τ, Data: D_m, Global model accuracy: A, Local model accuracy: a

**Output:** Local personalized models P_m, m ∈ [M]

1. **Cloud Server Side**
2. Initialize model ω^0
3. **for** t = 1 to T **do**
4.   τ = A
5.   **if** a < τ **then**
6.     Add device m to subset of edge device S
7.     Send local model ω^t to all selected edge device S
8.     **for** m = 1 to S **do**
9.       ω_m^{t+1}, a_m ← Local update (m, ω^t)
10.    ω^{t+1} ← ∑_{m∈M} ω^{t+1} / |S|
11.    A ← ∑_{m∈M} a_m / |M|

### Algorithm 2: PerFed-SKD: Local Model Training

**Input:** Edge device: m, Global model parameter: ω^t, Historical personalized model: V_m, Local dataset: D_m

**Output:** Local model parameter: ω_m^t, Local model accuracy: a

1. **Edge Device Side**
2. **for** each edge device m ∈ M **do in parallel**
3.   Receive global model parameter ω^t from the cloud server
4.   **if** m ∈ S **then**
5.     ω_m^t ← ω^t
6.   **else**
7.     ω_m^t ← ω_m^t
8.   **end if**
9.   **for** e = 1 to E **do**
10.    Compute local update using equation (3)
11.    ω_m^t ← ω_m^t - η∇φ_m(ω_m^t)
12.    Save personalized model V_m ← ω_m^t
13. Return ω_m^t, a

The proposed PerFed-SKD method is mainly focused on the global side and the major portion of the whole mechanism like training the teacher model on a large dataset, device selection, accuracy comparison, and aggregation, so the rounds of model parameters transmission between the edge device and the cloud server is reduced. PerFed-SKD also considers the nature of resource-constraint edge devices by reducing unnecessary training with the global model. Thus, the proposed framework is applicable for real-time applications more effectively.

---

## V. Experimental Analysis

In this section, we mention the details of the experimental setup followed by different dataset collections, and data partitioning techniques. We further analyze the performance of the proposed PerFed-SKD method with the standard FL models, discussed in the following subsections.

### A. Experimental Setup

The simulation is performed on a Dell Vostro workstation (CPU: 12th Gen Intel Core i7 - 12700). We have built an FL simulation setup in the Spyder integrated development environment where we have implemented all the models with the help of the PyTorch framework. We have used multiple scenarios by varying 20 to 80 edge devices with varied participation ratios. At the global level, 200 iterations were performed, while 20 epochs were run locally. The local learning rate η has been set to 0.01. The entire procedure is repeated ten times. The average accuracy and loss made by the edge devices are recorded in each iteration.

### B. Dataset and Evaluation Metrics

We use two popular and classical datasets in the computer vision era, i.e., MNIST [19] and EMNIST [20].

**Table II: Datasets Statistics**

| Dataset | Samples | Class | Task |
|---------|---------|-------|------|
| MNIST | 70,000 | 10 | Image classification |
| EMNIST | 60,000 | 26 | Image classification |

MNIST (Modified National Institute of Standards and Technology) dataset is related to computer vision. The MNIST dataset contains a collection of 70,000 handwritten digits (0-9) represented as 28×28 grayscale images. It is popular for benchmarking image classification algorithms, especially for digit recognition tasks. The dataset is split into 60,000 training images and 10,000 testing images. The EMNIST dataset is an extension of the MNIST dataset, which includes both handwritten digits and handwritten letters (uppercase and lowercase) from the Latin alphabet. The EMNIST dataset contains 814,255 images, with 47 balanced classes representing 26 uppercase and 26 lowercase letters, and 10 digits. Similar to MNIST, EMNIST images are represented as 28×28 grayscale images. The statistics of the datasets are depicted in Table II.

The performance of the proposed PerFed-SKD model is analyzed in terms of various evaluation metrics such as Accuracy, global loss, personalized local loss, user training time, and server aggregation time.

### C. Performance Analysis and Discussion

We compare PerFed-SKD with standard FL methods:

1. **FedAvg:** A distributed learning algorithm that trains models using decentralized data by aggregating local model updates from multiple clients using a weighted average [21].
2. **FedProx:** A variant of FedAvg that adds a regularization term to the objective function and encourages local models to be similar to the global model [4].
3. **Fed-ensemble:** Predicts by averaging K models updated by random permutations. Fed-ensemble is used in FL techniques without a computational overhead because it only sends one of the K models to edge devices in each communication round [22].
4. **PerFedAvg:** Federated training of deep feedforward neural networks using a base and personalization layer mitigates the negative effects of statistical heterogeneity [9].

We compared the above-mentioned algorithms with the PerFed-SKD method using the following evaluation metrics:

**Average Accuracy:** Table III represents the comparative analysis of the proposed PerFed-SKD and the state-of-the-art methods in terms of personalized average test accuracy under the Dirichlet distribution α = 0.001. PerFed-SKD consistently outperforms across different datasets and edge device setups. PerFed-SKD performs better than the best state-of-the-art method (PerFedAvg) by 5.46%. The effectiveness of PerFed-SKD on a large FL system suggests that the proposed model is scalable, which is crucial for implementation at the edge for handling real-time applications.

**Table III: Personalized Accuracy Using Dirichlet Distribution α = 0.001**

| Datasets | Scale | FedAvg | FedProx | PerFedAvg | FedEnsemble | PerFed-SKD |
|----------|-------|--------|---------|-----------|-------------|------------|
| MNIST | 20 Edge device | 91.1 ± 0.06 | 90.72 ± 0.04 | 92.5 ± 0.04 | 90.88 ± 0.07 | 95.69 ± 0.03 |
| MNIST | 80 Edge device | 88.67 ± 0.43 | 87.57 ± 0.36 | 89.12 ± 0.13 | 87.24 ± 0.06 | 92.47 ± 0.06 |
| EMNIST | 20 Edge device | 68.2 ± 0.86 | 68.4 ± 0.64 | 69.00 ± 0.37 | 69.15 ± 0.43 | 78.63 ± 0.46 |
| EMNIST | 80 Edge device | 65.89 ± 0.80 | 65.33 ± 0.46 | 67.45 ± 0.38 | 65.79 ± 0.35 | 73.57 ± 0.24 |

**Communication Efficiency:** Fig. 3 represents the personalized accuracy of the proposed PerFed-SKD and the state-of-the-art methods in each round over various datasets.

**Fig. 3. Average test accuracy of PerFed-SKD and state-of-the-art methods on various datasets: (a) MNIST and (b) EMNIST.**

For the MNIST dataset, we observe that the average accuracy achieved by FedAvg, FedProx, PerFedAvg, FedEnsemble, and PerFed-SKD are 90.86%, 90.45%, 91.89%, 90.88%, and 95.24% respectively. For the EMNIST dataset, we observe that the average accuracy achieved by FedAvg, FedProx, PerFedAvg, FedEnsemble, and PerFed-SKD are 68.23%, 68.16%, 69.12%, 69.15%, and 76.23% respectively. The proposed PerFed-SKD method consistently performs better than other baselines and exhibits a faster convergence. In its early phases, the performance growth rate of the PerFed-SKD method is almost identical to that of FedEnsemble. But with SKD, the proposed PerFed-SKD method later produces a better outcome. Further, Fig. 3 depicts that the proposed PerFed-SKD method requires the least training rounds to achieve higher accuracy.

**Average Loss:** The main target of PFL is to minimize losses while maintaining a reasonable average test accuracy. Lower losses represent uniform performance distribution of all local edge devices and make the proposed method fair one.

**Fig. 4. Average test loss of proposed PerFed-SKD and state-of-the-art methods on various datasets: (a) MNIST and (b) EMNIST.**

Fig. 4 shows the test loss in the training rounds across different datasets. For the MNIST dataset, we observe that the average loss achieved by FedAvg, FedProx, PerFedAvg, FedEnsemble, and PerFed-SKD are 2.13%, 2.15%, 2.17%, 5.42%, and 2.12% respectively. For the EMNIST dataset, we observe that the average loss achieved by FedAvg, FedProx, PerFedAvg, FedEnsemble, and PerFed-SKD are 3.16%, 3.17%, 3.15%, 13.15%, and 3.13% respectively. From this, we can observe that the PerFed-SKD method achieves a lower average loss than the baseline algorithms.

**Impacts of Participation Ratio:** As shown in Table IV, we considering various participation ratios r among 20%, 60%, 100% for each training round on the MNIST dataset. Regardless of the selection for the participation ratio, the proposed PerFed-SKD method consistently outperforms all baselines. The accuracy of all models increases when r grows from 20% to 60% since there are more training rounds for every edge device. When r increases from 60% to 100% some of the models including the PerFed-SKD method perform worse due to the possibility of model overfitting in edge devices.

**Table IV: Accuracy Using Various Participation Ratios for the MNIST Dataset**

| Models | r = 20% | r = 60% | r = 100% |
|--------|---------|---------|----------|
| FedAvg | 90.35 ± 1.08 | 89.64 ± 0.93 | 91.46 ± 0.43 |
| FedProx | 90.54 ± 1.28 | 90.42 ± 0.03 | 85.42 ± 0.03 |
| PerFedAvg | 90.65 ± 0.01 | 90.32 ± 0.83 | 91.32 ± 0.83 |
| FedEnsemble | 90.57 ± 3.64 | 90.12 ± 0.33 | 90.82 ± 0.63 |
| PerFed-SKD | 91.04 ± 0.28 | 93.22 ± 0.53 | 92.22 ± 0.33 |

**Effects of Data Heterogeneity:** By changing the concentration parameter of the Dirichlet distribution α as 0.001, 0.01, and 0.1, we changed the level of statistical heterogeneity. Table V shows that the proposed method consistently obtains the highest accuracy in non-IID settings. The proposed method still has a distinct advantage using balanced data distribution with α = 0.1 while other state-of-the-art PFL methods degrade their performance. The studies demonstrate the robustness and usefulness of PerFed-SKD regarding the degree of data heterogeneity.

**Table V: Accuracy With Different Data Heterogeneity on MNIST Dataset**

| Models | α = 0.001 | α = 0.01 | α = 0.1 |
|--------|-----------|----------|---------|
| FedAvg | 89.35 ± 0.48 | 88.64 ± 0.43 | 90.46 ± 0.33 |
| FedProx | 90.54 ± 0.28 | 89.42 ± 0.53 | 86.42 ± 0.73 |
| PerFedAvg | 90.65 ± 0.51 | 89.32 ± 0.83 | 90.32 ± 0.33 |
| FedEnsemble | 90.57 ± 0.64 | 89.12 ± 0.43 | 89.82 ± 0.23 |
| PerFed-SKD | 91.54 ± 0.78 | 92.52 ± 0.23 | 91.42 ± 0.93 |

**Fairness Among Edge Devices:** The personalized performance of different edge devices may vary with variations in the data division. Here, we analyze the performance of the PFL model of the local edge devices to determine the improvements in terms of personalized performance. The introduction to the SKD logic helps in achieving lightweight models that make it easier to compute and train in comparison to the heavy-weight models. Moreover, the computation of lightweight models requires less computing power, which helps the resource constraint devices to train the model efficiently and quickly. Table VI represents the Standard Deviation (SD) across all edge devices for the 20-scale scenario. A lower SD determines that the model is more equitable and the performance distributions of the edge devices are more consistent.

**Table VI: Average Accuracy and Standard Deviation of Different FL Methods Over Various Datasets**

| Models | MNIST | EMNIST |
|--------|-------|--------|
| FedAvg | 90.30 ± 5.08 | 67.42 ± 5.93 |
| FedProx | 89.30 ± 7.28 | 67.42 ± 5.03 |
| PerFedAvg | 87.65 ± 5.01 | 66.32 ± 4.83 |
| FedEnsemble | 89.57 ± 3.64 | 65.12 ± 6.33 |
| PerFed-SKD | 91.64 ± 3.88 | 75.22 ± 7.53 |

**Average User Training Time:**

**Fig. 5. Comparison of average user time of PerFed-SKD and standard models.**

In Fig 5, we observe the amount of time taken by the user in each training round along with communication time, local training time, and processing time in the server (model aggregation and distribution). The average user training time taken by FedAvg, FedProx, PerFedAvg, FedEnsemble, and PerFed-SKD are 51.34s, 50.48s, 52.03s, 51.14s, and 49.58s respectively for the MNIST dataset. For the EMNIST dataset, the average user training time taken by FedAvg, FedProx, PerFedAvg, FedEnsemble, and PerFed-SKD are 68.12s, 67.23s, 68.34s, 69.54s, and 64.27s respectively. From this, we can observe that the average user training time is reduced in the case of the PerFed-SKD method in comparison to the state-of-the-art methods. The proposed PerFed-SKD method achieves this improvement due to the introduction of the threshold value concept that helps to reduce the user training time. To achieve higher accuracy, the total training time is calculated as the multiplication of the entire training round and the training time per round. Although the proposed PerFed-SKD method requires little time overhead in each training round, however, it achieves higher convergence speed while achieving targeted accuracy.

The proposed PerFed-SKD method has two important targets. First, the method can store the past predictions of the personalized model v_m and recompute the predictions when local training. The other is the selection of edge devices for local training in each round to reduce the communication overhead. We achieved our goals by applying SKD with aggregated accuracy comparison for selecting devices and found the proposed method is better than existing ones.

---

## VI. Conclusion

In this paper, we have presented the design, implementation, and evaluation of the proposed PerFed-SKD, a PFL framework that significantly improves communication overhead and achieves personalization under data heterogeneity. We examined personalized knowledge-forgetting phenomena by analyzing standard FedAvg and FedProx algorithms. The proposed method fully utilizes historical personalized models through SKD that achieve a better balance between generalization and personalization while reducing losses. By applying PerFed-SKD, each edge device learns with the current model as well as with previous saved knowledge, as opposed to a globally shared model. Extensive simulation results demonstrated the outperformance of the proposed PerFed-SKD over state-of-the-art methods, i.e., two standard non-PFLs (FedAvg, FedProx) and two PFLs (FedEnsemble, PerFedAvg) methods using publicly available image datasets. In the future, we would like to implement a semi-synchronous mechanism in edge networks that can reduce the overall training time and increase communication efficiency.

---

## References

[1] X. Sun, M. Wang, R. Lin, Y. Sun, and S. Shin Cheng, "Deep-learned perceptual quality control for intelligent video communication," *IEEE Trans. Consum. Electron.*, vol. 68, no. 4, pp. 354-365, Nov. 2022.

[2] S. Basu, D. Bera, and S. Karmakar, "Detection and intelligent control of cloud data location using hyperledger framework," *IEEE Trans. Consum. Electron.*, vol. 69, no. 1, pp. 76-86, Feb. 2023.

[3] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, "Federated learning with non-IID data," 2018, *arXiv:1806.00582*.

[4] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in *Proc. Mach. Learn. Syst.*, vol. 2, pp. 429-450, 2020.

[5] A. Asesh, "Federated learning: Optimizing objective function," in *Proc. IEEE Int. Conf. Artif. Intell. Eng. Technol. (IICAIET)*, 2021, pp. 1-6.

[6] C. T. Dinh, N. Tran, and J. Nguyen, "Personalized federated learning with moreau envelopes," *Adv. Neural Inf. Process. Syst.*, vol. 33, pp. 21394-21405, Dec. 2020.

[7] Y. Qin and M. Kondo, "MLMG: Multi-local and multi-global model aggregation for federated learning," in *Proc. IEEE Int. Conf. Pervasive Comput. Commun. Workshops Other Aff. Events (PerCom Workshops)*, 2021, pp. 565-571.

[8] A. Li et al., "LotteryFL: Empower edge intelligence with personalized and communication-efficient federated learning," in *Proc. IEEE/ACM Symp. Edge Comput. (SEC)*, 2021, pp. 68-79.

[9] A. Fallah, A. Mokhtari, and A. Ozdaglar, "Personalized federated learning: A meta-learning approach," 2020, *arXiv:2002.07948*.

[10] Y. Jiang, J. Konečný, K. Rush, and S. Kannan, "Improving federated learning personalization via model agnostic meta learning," 2019, *arXiv:1909.12488*.

[11] X. Qu et al., "Personalized federated learning for heterogeneous residential load forecasting," *Big Data Mining Anal.*, vol. 6, no. 4, pp. 421-432, Dec. 2023.

[12] Y. Jiang, X. Lu, H. Zheng, and W. Mao, "ASPFL: Efficient Personalized federated learning for edge based on adaptive sparse training," in *Proc. IEEE Int. Conf. Web Services (ICWS)*, 2023, pp. 269-277.

[13] L. Zhang and Z. Xu, "K-PFed: Communication-efficient personalized federated clustering," in *Proc. IEEE 3rd Int. Conf. Electron. Technol., Communication Inf. (ICETCI)*, 2023, pp. 1026-1029.

[14] C. Li, G. Li, and P. K. Varshney, "Decentralized federated learning via mutual knowledge transfer," *IEEE Internet Things J.*, vol. 9, no. 2, pp. 1136-1147, Jan. 2022.

[15] P. P. Liang, T. Liu, L. Ziyin, R. Salakhutdinov, and L.-P. Morency, "Think locally, act globally: Federated learning with local and global representations," 2020, *arXiv:2001.01523*.

[16] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," 2015, *arXiv:1503.02531*.

[17] L. Zhang, J. Song, A. Gao, J. Chen, C. Bao, and K. Ma, "Be your own teacher: Improve the performance of convolutional neural networks via self distillation," in *Proc. IEEE/CVF Int. Conf. Comput. Vis.*, 2019, pp. 3713-3722.

[18] K. Kim, B. Ji, D. Yoon, and S. Hwang, "Self-knowledge distillation with progressive refinement of targets," in *Proc. IEEE/CVF Int. Conf. Comput. Vis.*, 2021, pp. 6567-6576.

[19] Y. LeCun, C. Cortes, and C. Burges, "MNIST handwritten digit database," *ATT Labs*. [Online]. Available: http://yann.lecun.com/exdb/mnist, vol. 2, 2010.

[20] G. Cohen, S. Afshar, J. Tapson, and A. van Schaik, "EMNIST: Extending MNIST to handwritten letters," in *Proc. Int. Joint Conf. Neural Netw. (IJCNN)*, 2017, pp. 2921-2926.

[21] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Y. Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. Artif. Intell. Statist.*, PMLR, 2017, pp. 1273-1282.

[22] N. Shi, F. Lai, R. A. Kontar, and M. Chowdhury, "Fed-ensemble: Ensemble models in federated learning for improved generalization and uncertainty quantification," *IEEE Trans. Autom. Sci. Eng.*, early access, May 1, 2023, doi: 10.1109/TASE.2023.3269639.