
## IV. CASE STUDIES

### A. Experimental Setups

1) **Dataset Description**: To validate our proposed method, we test our model on two publicly available datasets, REFIT [36] and REDD [37]. REFIT contains appliance-level power consumption of 20 houses in the U.K., where every house is installed with 10 power sensors comprising a current clamp for the household aggregate and 9 individual appliance monitors (IAMs) with a time resolution of 8 seconds. REDD consists of whole-home and appliance-specific electricity consumption for 6 real houses in the USA with a time resolution of 3 seconds, where each of these houses contains 9 to 24 different appliances. Our case study follows the setup of previous research [29], where a small amount of pre-collected labeled appliance data is used for model training.

2) **Model Settings**: We use 5 benchmarks to illustrate the strength of our proposed models. The centralized model is trained using all data from different households by assuming no privacy concerns. The local model is trained solely using data collected by the end device for a single household. NAS and MNAS perform an architecture search before full model training on single household data. The federated model employs FedAvg [38] across households.

Our framework is a basic one that can be compatible with any state-of-the-art model and network used in general NILM tasks. In our experiments, we choose one-dimensional CNN, which is of interest in NILM, as the backbone. The fixed architecture for non-search models involves five convolutional layers with a 7-size kernel, followed by one dense layer. The superiority of the search space can be assessed by space diversity, efficiency evaluation, and comparative analysis. To find the optimal architecture, we carefully design a search space, which consists of 5 nodes and contains 8 candidate operations for each path, including {conv_7, conv_5, conv_3, conv_1, max pooling, average pooling, skip and identity}. The developed search space with enriched architecture types has been validated to yield device-friendly memory overhead and achieve significant performance on all appliances and households. After the search phase, the first two paths with the largest architecture parameters are retained for each node. Taking into account the trade-off between model communication and model representation capacity, the proxy models are composed of three convolutional layers with a kernel size of 5 and a single dense layer. The code for the experiments has been publicly available.

3) **Evaluation Metrics**: We use two metrics to measure the prediction accuracy for a single appliance: mean absolute error (MAE) and signal absolute error (SAE). MAE measures the average deviation between the estimates \(\hat{y}_t\) and true values \(y_t\) and SAE measures the deviation between the estimated total energy consumption and true consumption. The expressions are as follows:

\[\begin{array}{l}\mathrm{MAE} = \frac{1}{T}\sum_{t = 1}^{T}|y_t - \hat{y}_t|\\ \mathrm{SAE} = \frac{\left|\sum_{t = 1}^{T}\hat{y}_t - \sum_{t = 1}^{T}y_t\right|}{\sum_{t = 1}^{T}y_t} \end{array} \quad (20)\]

*[Insert Fig. 5. Load disaggregation results for fridge, washing machine, dish washer, microwave, and kettle]*

### B. Basic Results

Fig. 5 displays the predicted consumption for five different appliances, with distinct patterns observed. The power consumption of fridges exhibits a periodic pattern as they operate on a cooling cycle, where the compressor turns on periodically to cool the interior. Washing machines and dish washers are typically used continuously over a certain period, leading to sustained energy consumption. Conversely, consumption patterns for appliances like microwaves and kettles are typically abrupt and short-lived. Our proposed model captures the periodic usage pattern with high precision. Our model predicts consumption with minor deviations for aperiodic usage patterns with varying time durations. The different usage patterns are also reflected in the searched architectures, as shown in Fig. 6. For the fridge, the architecture of NAS includes few convolutional layers because cyclic power consumption patterns are easily identified. By contrast, the searched network for the dish washer integrates more convolutional layers to capture continuously varying power consumption characteristics. For the microwave, the personalized model contains several pooling layers to focus on localized information of short duration.

*[Insert Fig. 6. Illustration of the searched architectures for different appliances]*

**TABLE I COMPARISON OF MODEL PERFORMANCE ON THE REFIT DATASET**

| Appliance | Metric | Centralized | Local | NAS | MNAS | Federated | Proposed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fridge** | MAE | 32.15 | 30.14 | 28.41 | 28.5 | 29.05 | **27.99** |
| | SAE | 1.131 | 1.125 | 1.042 | 1.057 | 1.083 | **1.034** |
| **Washing machine** | MAE | 18.25 | 23.74 | 19.55 | 19.05 | 21.94 | **17.34** |
| | SAE | 1.508 | 1.781 | 1.742 | 1.525 | 1.821 | **1.460** |
| **Dish washer** | MAE | 38.42 | 39.36 | 36.94 | 36.99 | 36.58 | **36.11** |
| | SAE | 1.176 | 1.223 | 1.167 | 1.173 | 1.14 | **1.133** |
| **Microwave** | MAE | 9.52 | 10.22 | 9.24 | 9.49 | 9.56 | **8.58** |
| | SAE | 1.496 | 1.797 | 1.498 | 1.521 | 1.485 | **1.401** |
| **Kettle** | MAE | 20.64 | 23.25 | 21.03 | 20.81 | 20.75 | **18.24** |
| | SAE | 1.813 | 2.012 | 1.796 | 1.789 | 1.856 | **1.552** |

**TABLE II COMPARISON OF MODEL PERFORMANCE ON THE REDD DATASET**

| Appliance | Metric | Centralized | Local | NAS | MNAS | Federated | Proposed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fridge** | MAE | 32.78 | 34.45 | 32.83 | 32.47 | 32.99 | **31.73** |
| | SAE | 0.475 | 0.496 | 0.478 | 0.471 | 0.476 | **0.462** |
| **Dish washer** | MAE | 11.74 | 9.63 | 9.06 | 8.84 | 8.88 | **8.51** |
| | SAE | 1.265 | 1.121 | 1.107 | 1.097 | 1.052 | **1.008** |
| **Microwave** | MAE | 18.74 | 20.52 | 19.22 | 18.96 | 18.31 | **17.89** |
| | SAE | 1.126 | 1.22 | 1.158 | 1.125 | 1.098 | **1.081** |

We compare the performance of our proposed model with benchmark models using the REFIT and REDD datasets. The performance of the REFIT and REDD are summarized in Table I and Table II respectively with the best model highlighted in bold. In both two datasets, our proposed model achieves the best performance with MAE and SAE for all appliances. Using the local model as the baseline case, we can observe that the centralized model outperforms the local model for most appliances by acquiring data from other households. Both NAS and MNAS models outperform the local model by introducing architecture search before the model training stages. Notably, MNAS yields comparable and even higher accuracy to NAS on the REDD dataset. The possible reasons include: 1) the efficiency of the compressed search space enhances model generalization, 2) the focusedness of the single-path search strategy accelerates model convergence, and 3) the regularization of the hardware-aware evaluation mitigates model overfitting. A detailed comparison of the model efficiency will be carried out in later sections. The federated model without architecture search shows better performance compared to the local model across most appliances. This is expected since federated learning empowers the local model with knowledge from other households while preserving privacy. Overall, the local model shows the lowest accuracy with only a few exceptions where appliance usage patterns are better captured using only local data. Our proposed model, uniting the strength of architecture search and federation, shows the best performance.

### C. Model Effectiveness

1) **Performance on Different Appliances**: To clearly and intuitively demonstrate the enhancements brought by the proposed method in terms of energy assessment and monetary gain, we further incorporate the MAE in units of kWh as an additional metric in our experiment. We summarize the relative improvement of on-device models compared with the base local model over the REFIT and REDD datasets in Fig. 7. The centralized model is excluded since it can not be implemented on end devices. We can observe a larger improvement in the washing machine, microwave, and kettle with over \(10\%\) MAE enhancement for the proposed model. The improvement for the fridge is not significant, which may be due to the simplicity of its prediction task. Both the NAS and the MNAS models have a competitive edge over the federated model for the fridge and washing machine, indicating that local knowledge is effective in these appliances. For the dish washer, global knowledge contributes to better prediction accuracy. Our proposed model combines the MNAS and federated learning approaches, resulting in an overall improvement in energy consumption estimation. This highlights the benefits of harnessing both local and global knowledge.

*[Insert Fig. 7. Relative improvements in energy consumption estimation of different methods for different appliances]*

2) **Performance on Different Households**: Fig. 8 shows the relative improvement for the baseline of different households in the REFIT dataset. In general, all houses benefit from the proposed method with universal performance improvement. However, the other on-device models show a minor degradation on some specific houses, such as House 6. This is because House 6 owns 49 appliances, which is the most among other households. Excessive appliance loads complicate the disaggregation task by bringing more uncertainty. In contrast, by integrating local and global knowledge in our model, the aforementioned underperforming House 6 can have a better performance than the baseline model. In addition, Houses 2, 4, and 19 show an improving accuracy using NAS and MNAS and degrading performance using federated learning. Such households have distinctive consumption patterns and, hence, do not benefit from the inclusion of global knowledge. The accuracy improvement is more pronounced for houses 4 and 17 using MNAS compared to NAS, which may be attributed to the fact that the localized information better reflects their frequent total power variations, whereas MNAS focuses more on searching with smaller kernel-size convolutional layers in the compressed supernet. Fig. 9 shows the household-wise relative improvement for the REDD dataset. Our proposed model outperforms other benchmarks by showing improvement across all 6 households. Federated learning also shows a universal improvement, while NAS and MNAS show degraded accuracy when applied to House 4. This phenomenon can be explained by the fact that the REDD dataset contains a shorter period compared to the REFIT dataset, and therefore the inclusion of global information improves the model performance by solving the local data paucity problem.

*[Insert Fig. 8. Relative improvements of different methods for different houses on REFIT dataset]*
*[Insert Fig. 9. Relative improvements of different methods for different houses on REDD dataset]*

### D. Model Robustness

1) **Data Imbalance**: Large-scale end devices may exhibit varying amounts of data due to instances of device failure, device replacement, and data loss. This imbalance may drive the global model to favor devices with larger data volumes. We conducted experiments to explore the influence of data imbalance on the performance of our proposed method. The imbalance is generated by transferring \(30\%\) of the original data from a portion of the devices to the remaining devices. Table III demonstrates the MAE results with the imbalance ratio on the REFIT dataset. We can observe a certain growth in decomposition error as the imbalance ratio increases due to the high contribution of data volume to model generalization. Overall, the proposed method maintains satisfactory performance with no more than a \(4\%\) accuracy drop, except in extreme cases where half of the devices are affected by data imbalance issues.

**TABLE III MODEL PERFORMANCE WITH DIFFERENT IMBALANCE RATIO**

| Appliance | Imbalance ratio 0% | 10% | 25% | 50% |
| :--- | :--- | :--- | :--- | :--- |
| Fridge | 27.99 | 28.13 | 28.35 | 29.59 |
| Washing machine | 17.34 | 18.05 | 18.86 | 20.23 |
| Disher washer | 36.11 | 36.49 | 36.63 | 37.78 |
| Microwave | 8.58 | 8.89 | 9.01 | 9.59 |
| Kettle | 18.24 | 18.97 | 19.41 | 20.44 |
| **Average** | **21.65** | **22.10** | **22.45** | **23.52** |

2) **Device Heterogeneity**: Large-scale end devices may encounter varying local training durations due to differences in real-time task occupancy, network communication velocity, and operating frequency configuration. This heterogeneity could potentially cause asynchronous uploading of model weights to the central server. We carried out experiments to investigate the effect of device heterogeneity on the performance of our proposed method. The heterogeneity is introduced by prolonging the training time of a portion of the devices to twice the original duration. Table IV presents the MAE results with the heterogeneity ratio on the REFIT dataset. Remarkably, asynchronous aggregation with insignificant heterogeneity can almost match or even surpass the accuracy of training in a non-heterogeneity environment for certain appliances. Even though half of the devices experience delays, the proposed method still achieves stable accuracy, demonstrating significant robustness against device heterogeneity.

**TABLE IV MODEL PERFORMANCE WITH DIFFERENT HETEROGENEITY RATIO**

| Appliance | Heterogeneity ratio 0% | 10% | 25% | 50% |
| :--- | :--- | :--- | :--- | :--- |
| Fridge | 27.99 | 28.05 | 28.14 | 28.40 |
| Washing machine | 17.34 | 17.36 | 17.61 | 18.19 |
| Disher washer | 36.11 | 36.05 | 36.73 | 37.38 |
| Microwave | 8.58 | 8.67 | 8.83 | 9.05 |
| Kettle | 18.24 | 18.40 | 18.79 | 19.11 |
| **Average** | **21.65** | **21.70** | **22.02** | **22.43** |

### E. Complexity Analysis

To compare the model efficiency of NAS and the proposed method, we provide an analysis in terms of time complexity and space complexity. Let \(\mathcal{E}\) and \(\mathcal{O}\) denote the set of edges and candidate operations for each path, respectively. Additionally, let \(K_{i,j}\) \(M_{i,j}\) and \(C_j\) denote the kernel size, feature map length, and channel number of \(i\)-th operation on the \(j\)-th edge, respectively. For searching architecture with NAS, the time complexity can be expressed as Time \(\sim O(\sum_{j\in \mathcal{E}}\sum_{i\in \mathcal{O}}K_{i,j}\) \(M_{i,j}\cdot C_{j - 1}\cdot C_j)\) and the space complexity (i.e., memory footprint) can be expressed as \(Space\sim O(\sum_{j\in \mathcal{E}}\sum_{i\in \mathcal{O}}K_{i,j}\) \(C_{j - 1}\cdot C_j + M_{i,j}\cdot C_j)\) . In our approach, both the search space and the search strategy are improved to increase model efficiency. Specifically, only the largest convolution kernel needs to be stored, and only one path needs to be activated in memory per search step. Consequently, the time complexity can be expressed as Time \(\sim O(\sum_{j\in \mathcal{E}}K_{j}^{*}\cdot M_{j}^{*}\cdot C_{j - 1}\cdot C_j)\) and the space complexity can be expressed as \(Space\sim O(\sum_{j\in \mathcal{E}}K_{j}^{*}\cdot C_{j - 1}\cdot C_j + M_{j}^{*}\cdot C_j)\) . Here \(K_{j}^{*}\) and \(M_{j}^{*}\) denote the maximum kernel size and feature map length of \(j\)-th edge. Thus the proposed method can save roughly \(M\) times memory consumption on the end device compared to the NAS.

**TABLE V COMPARISON OF TESTING EFFICIENCY FOR DIFFERENT METHODS**

| Method | Model size (KB) | Inference time (s) |
| :--- | :--- | :--- |
| Local | 29.05 | 6.55E-03 |
| NAS | 10.97 | 4.18E-03 |
| Proposed | 7.01 | 2.53E-03 |

To evaluate the computational expense, we compare the training and testing efficiency of the non-search local model, the NAS model, and our proposed model. The training time and search time of three models for the fridge in the REDD dataset are shown in Fig. 10. It can be found that the proposed model remarkably reduces the search time and training time of NAS, bringing it to the same level as the local model. The model size and inference time for each sample are listed in Table V. The results show that the size of the customized architecture, i.e., the memory space required, is smaller than that of the fixed model. Prior research has examined the correlation between the memory required for model training and the model size, revealing an empirical disparity of approximately 20 times [39]. As such, the memory overhead necessary for compact model training can be estimated as 140.22 KB. This overhead is feasible for model deployment on widely used end devices, such as smart meters equipped with the STM32F405 microcontroller, which has a static random-access memory of 192KB. Moreover, the proposed hardware-aware method can limit disaggregating time to a preset sliding time window and reduce the latency time by more than 2.5 times. Note that the actual runtime on the end device will be longer than the above given time measured by the high-performance GPU device. Overall, the improved efficiency showcases the feasibility and effectiveness of our model for implementation on resource-constrained end devices.

*[Insert Fig. 10. Comparison of training time and search time]*

## V. CONCLUSION AND FUTURE WORKS

In this paper, we propose a lightweight federated learning approach to enable NILM on resource-constrained devices. The developed memory-efficient NAS algorithm reduces computation resource consumption to the same level as compact model training. The proposed method develops personalized models with unique network architecture customized to various appliances with distinct usage patterns, boosting decomposition accuracy. Furthermore, the federated framework integrated with adaptive mutual learning can utilize distributed data to collaboratively train heterogeneous models in a privacy-preserving manner. The global knowledge transferred from unified proxy models further enhances the generalization of personalized models, with better performance than benchmark models. The results show significant improvements over \(15\%\) for appliances such as washing machines, microwaves, and kettles on the REFIT dataset, and an overall improvement of around \(10\%\) for different appliances on the REDD dataset. Importantly, our proposed method achieves considerable improvements in memory footprint and computational time. Furthermore, the proposed method demonstrates robustness in the presence of imbalanced data and device heterogeneity. In brief, our method offers a promising and feasible solution for on-device NILM applications.

However, our work also presents the following limitations. First, the improved NAS still slightly increases the computational cost on end devices since they need to learn customized model architecture from the search space. Second, our study presumes that the communication network and the aggregation server are both reliable and secure, thereby overlooking potential threats posed by hackers and malicious attackers.

As illustrated in the case studies, even though lightweight federated learning outperforms the benchmarks overall, federated learning and NAS achieve the best performance on specific appliances and households. An intriguing area of study would be to analyze whether local and global knowledge is gainful on a given task. In addition, communication overhead is also a critical efficiency metric for end devices. Thus, another direction for future work will focus on combining the proposed approach with gradient quantization techniques to alleviate communication costs in practice.

## REFERENCES

[1] Z. Liu, Z. Deng, S. Davis, and P. Ciais, "Monitoring global carbon emissions in 2022," *Nature Rev. Earth Environ.*, vol. 4, pp. 205-206, Mar. 2023.
[2] P. A. Schirmer and I. Mporas, "Non-intrusive load monitoring: A review," *IEEE Trans. Smart Grid*, vol. 14, no. 1, pp. 769-784, Jan. 2023.
[3] H. Cimen, N. Cetinkaya, J. C. Vasquez, and J. M. Guerrero, "A microgrid energy management system based on non-intrusive load monitoring via multitask learning," *IEEE Trans. Smart Grid*, vol. 12, no. 2, pp. 977-987, Mar. 2021.
[4] B. Liu, W. Luan, J. Yang, and Y. Yu, "The balanced window-based load event optimal matching for NILM," *IEEE Trans. Smart Grid*, vol. 13, no. 6, pp. 4690-4703, Nov. 2022.
[5] W. Luan, R. Zhang, B. Liu, B. Zhao, and Y. Yu, "Leveraging sequence-to-sequence learning for online non-intrusive load monitoring in edge device," *Int. J. Elect. Power Energy Syst.*, vol. 148, Jun. 2023, Art. no. 108910.
[6] T.-T.-H. Le, S. Heo, and H. Kim, "Toward load identification based on the Hilbert transform and sequence to sequence long short-term memory," *IEEE Trans. Smart Grid*, vol. 12, no. 4, pp. 3252-3264, Jul. 2021.
[7] C. Zhang, M. Zhong, Z. Wang, N. Goddard, and C. Sutton, "Sequence-to-point learning with neural networks for non-intrusive load monitoring," in *Proc. AAAI Conf. Artif. Intell.*, 2018, pp. 2604-2611.
[8] P. A. Schirmer and I. Mporas, "Double fourier integral analysis based convolutional neural network regression for high-frequency energy disaggregation," *IEEE Trans. Emerg. Topics Comput. Intell.*, vol. 6, no. 3, pp. 439-449, Jun. 2022.
[9] J. Chen, X. Wang, X. Zhang, and W. Zhang, "Temporal and spectral feature learning with two-stream convolutional neural networks for appliance recognition in NILM," *IEEE Trans. Smart Grid*, vol. 13, no. 1, pp. 762-772, Jan. 2022.
[10] Y. Liu, J. Qiu, and J. Ma, "SAMNet: Toward latency-free non-intrusive load monitoring via multi-task deep learning," *IEEE Trans. Smart Grid*, vol. 13, no. 3, pp. 2412-2424, May 2022.
[11] D. Garcia-Perez, D. Perez-Lopez, I. Diaz-Blanco, A. Gonzalez-Muniz, M. Dominguez-Gonzalez, and A. A. C. Vega, "Fully-convolutional denoising auto-encoders for NILM in large non-residential buildings," *IEEE Trans. Smart Grid*, vol. 12, no. 3, pp. 2722-2731, May 2021.
[12] L. Wang, S. Mao, and R. M. Nelms, "Transformer for non-intrusive load monitoring: Complexity reduction and transferability," *IEEE Internet Things J.*, vol. 9, no. 19, pp. 18987-18997, Oct. 2022.
[13] M. D'Incecco, S. Squartini, and M. Zhong, "Transfer learning for non-intrusive load monitoring," *IEEE Trans. Smart Grid*, vol. 11, no. 2, pp. 1419-1429, Mar. 2020.
[14] Y. Han, K. Li, C. Wang, F. Si, and Q. Zhao, "Unknown appliances detection for non-intrusive load monitoring based on conditional generative adversarial networks," *IEEE Trans. Smart Grid*, vol. 14, no. 6, pp. 4553-4564, Nov. 2023.
[15] S. Mari, G. Bucci, F. Ciancetta, E. Fiorucci, and A. Fioravanti, "An embedded deep learning NILM system: A year-long field study in real houses," *IEEE Trans. Instrum. Meas.*, vol. 72, pp. 1-15, Oct. 2023.
[16] E. Tabanelli, D. Brunelli, A. Acquaviva, and L. Benini, "Trimming feature extraction and inference for MCU-based edge NILM: A systematic approach," *IEEE Trans. Ind. Informat.*, vol. 18, no. 2, pp. 943-952, Feb. 2021.
[17] Y. Liu, Q. Xu, Y. Yang, and W. Zhang, "Detection of electric bicycle indoor charging for electrical safety: A NILM approach," *IEEE Trans. Smart Grid*, vol. 14, no. 5, pp. 3862-3875, Sep. 2023.
[18] S. Ahmed and M. Bons, "Edge computed NILM: A phone-based implementation using MobileNet compressed by TensorFlow lite," in *Proc. 5th Int. Workshop Non-Intrusive Load Monit.*, 2020, pp. 44-48.
[19] S. Sykiotis et al., "Performance-aware NILM model optimization for edge deployment," *IEEE Trans. Green Commun. Netw.*, vol. 7, no. 3, pp. 1434-1446, Sep. 2023.
[20] Q. Luo, T. Yu, C. Lan, Y. Huang, Z. Wang, and Z. Pan, "A generalizable method for practical non-intrusive load monitoring via metric-based meta-learning," *IEEE Trans. Smart Grid*, vol. 15, no. 1, pp. 1103-1115, Jan. 2024.
[21] S. Athanasoulias, S. Sykiotis, M. Kaselimi, A. Doulamis, N. Doulamis, and N. Ipiotis, "OPT-NILM: An iterative prior-to-full-training pruning approach for cost-effective user side energy disaggregation," *IEEE Trans. Consum. Electron.*, vol. 70, no. 1, pp. 4435-4446, Feb. 2024.
[22] Y. Zhang et al., "FedNILM: Applying federated learning to NILM applications at the edge," *IEEE Trans. Green Commun. Netw.*, vol. 7, no. 2, pp. 857-868, Jun. 2023.
[23] B. Zoph and Q. V. Le, "Neural architecture search with reinforcement learning," 2017, *arXiv:1611.01578*.
[24] H. Pham, M. Guan, B. Zoph, Q. Le, and J. Dean, "Efficient neural architecture search via parameters sharing," in *Proc. Int. Conf. Mach. Learn.*, 2018, pp. 4095-4104.
[25] H. Liu, K. Simonyan, and Y. Yang, "DARTS: Differentiable architecture search," 2018, *arXiv:1806.09055*.
[26] H. Cai, L. Zhu, and S. Han, "ProxylessNAS: Direct neural architecture search on target task and hardware," 2018, *arXiv:1812.00332*.
[27] "General data protection regulation," 2018. [Online]. Available: https://gdpr-info.eu/
[28] S. Lee and D.-H. Choi, "Federated reinforcement learning for energy management of multiple smart homes with distributed energy resources," *IEEE Trans. Ind. Informat.*, vol. 18, no. 1, pp. 488-497, Jan. 2022.
[29] H. Wang, C. Si, G. Liu, J. Zhao, F. Wen, and Y. Xue, "Fed-NILM: A federated learning-based non-intrusive load monitoring method for privacy-protection," *Energy Convers. Econ.*, vol. 3, no. 2, pp. 51-60, 2022.
[30] T. Wang and Z. Dong, "Blockchain-based clustered federated learning for non-intrusive load monitoring," *IEEE Trans. Smart Grid*, vol. 15, no. 2, pp. 2348-2361, Mar. 2024.
[31] S. Dai, F. Meng, Q. Wang, and X. Chen, "DP2-NILM: A distributed and privacy-preserving framework for non-intrusive load monitoring," *Renew. Sustain. Energy Rev.*, vol. 191, Mar. 2024, Art. no. 114091.
[32] Q. Li, J. Ye, W. Song, and Z. Tse, "Energy disaggregation with federated and transfer learning," in *Proc. IEEE 7th World Forum Internet Things (WF-IoT)*, 2021, pp. 698-703.
[33] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. Artif. Intell. Statist.*, 2017, pp. 1273-1282.
[34] F. Ciancetta, G. Bucci, E. Fiorucci, S. Mari, and A. Fioravanti, "A new convolutional neural network-based system for NILM applications," *IEEE Trans. Instrum. Meas.*, vol. 70, pp. 1-12, Nov. 2020.
[35] E. Jang, S. Gu, and B. Poole, "Categorical reparameterization with Gumbel-Softmax," 2016, *arXiv:1611.01144*.
[36] D. Murray, L. Stankovic, and V. Stankovic, "REFIT: Electrical load measurements (cleaned)," Dataset, Univ. Strathclyde, PURE, Glasgow, U.K., 2016.
[37] J. Z. Kolter and M. J. Johnson, "REDD: A public data set for energy disaggregation research," in *Proc. Workshop Data Min. Appl. Sustain. (SIGKDD)*, San Diego, CA, USA, 2011, pp. 59-62.
[38] X. Li, K. Huang, W. Yang, S. Wang, and Z. Zhang, "On the convergence of FedAvg on non-IID data," 2019, *arXiv:1907.02189*.
[39] N. S. Sohoni, C. R. Aberger, M. Leszczynski, J. Zhang, and C. Ré, "Low-memory neural network training: A technical report," 2019, *arXiv:1904.10631*.

---
**Yehui Li** (Graduate Student Member, IEEE) received the B.S. degree in electronic science and technology from the Harbin Institute of Technology in 2022. He is currently pursuing the Ph.D. degree in electrical and electronic engineering with The University of Hong Kong. His current research interests include data analytics and edge intelligence in smart grids.

**Ruiyang Yao** (Graduate Student Member, IEEE) received the M.Math. degree in mathematics and statistics from the University of Oxford and the M.S. degree in computing from Imperial College London. He is currently pursuing the Ph.D. degree in electrical and electronic engineering with The University of Hong Kong. His current research interests include data analytics and data security in smart grids.

**Dalin Qin** (Graduate Student Member, IEEE) received the B.S. degree in electrical engineering and its automation from the South China University of Technology, Guangzhou, China, in 2022. He is currently pursuing the Ph.D. degree in electrical and electronic engineering with The University of Hong Kong. His current research interests include energy forecasting and privacy-preserving data analytics in smart grids.

**Yi Wang** (Senior Member, IEEE) received the B.S. degree from the Huazhong University of Science and Technology in June 2014, and the Ph.D. degree from Tsinghua University in January 2019. He was a visiting student with the University of Washington from March 2017 to April 2018. He served as a Postdoctoral Researcher with the Power Systems Laboratory, ETH Zurich from February 2019 to August 2021. He is currently an Assistant Professor with the Department of Electrical and Electronic Engineering, The University of Hong Kong. His research interests include data analytics in smart grids, energy forecasting, multienergy systems, Internet-of-Things, and cyber-physical-social energy systems.