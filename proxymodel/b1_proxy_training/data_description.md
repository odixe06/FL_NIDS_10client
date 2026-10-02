# Task 2: Data preprocesseing

[Bản cũ](https://www.notion.so/B-n-c-34c730cee68d80f29c59dae5b77bc44d?pvs=21)

**Giới thiệu về dataset CIC IOT:** 

[Task 1: Dataset CIC IOT 2023](https://www.notion.so/Task-1-Dataset-CIC-IOT-2023-321730cee68d804fb514d5ad2ed951ec?pvs=21)

# 1. Thống kê dataset gốc:

| Feature | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flow_duration | 5.76544939 | 285.034171 | 0 | 0 | 0 | 0.10513809 | 394357.207 |
| Header_Length | 76705.9637 | 461331.747 | 0 | 54 | 54 | 280.555 | 9907147.75 |
| Protocol type | 9.06568989 | 8.94553292 | 0 | 6 | 6 | 14.33 | 47 |
| Duration | 66.3507169 | 14.0191881 | 0 | 64 | 64 | 64 | 255 |
| Rate | 9064.05724 | 99562.4906 | 0 | 2.09185589 | 15.7542308 | 117.384754 | 8388608 |
| Srate | 9064.05724 | 99562.4906 | 0 | 2.09185589 | 15.7542308 | 117.384754 | 8388608 |
| Drate | 5.46E-06 | 0.00725077 | 0 | 0 | 0 | 0 | 29.7152249 |
| fin_flag_number | 0.08657207 | 0.28120696 | 0 | 0 | 0 | 0 | 1 |
| syn_flag_number | 0.20733528 | 0.40539779 | 0 | 0 | 0 | 0 | 1 |
| rst_flag_number | 0.09050473 | 0.28690351 | 0 | 0 | 0 | 0 | 1 |
| psh_flag_number | 0.08775006 | 0.28293106 | 0 | 0 | 0 | 0 | 1 |
| ack_flag_number | 0.12343168 | 0.32893207 | 0 | 0 | 0 | 0 | 1 |
| ece_flag_number | 1.48E-06 | 0.00121571 | 0 | 0 | 0 | 0 | 1 |
| cwr_flag_number | 7.28E-07 | 0.00085338 | 0 | 0 | 0 | 0 | 1 |
| ack_count | 0.09054283 | 0.28643144 | 0 | 0 | 0 | 0 | 7.7 |
| syn_count | 0.33035785 | 0.6635354 | 0 | 0 | 0 | 0.06 | 12.87 |
| fin_count | 0.09907672 | 0.32711642 | 0 | 0 | 0 | 0 | 248.32 |
| urg_count | 6.23982356 | 71.8524536 | 0 | 0 | 0 | 0 | 4401.7 |
| rst_count | 38.4681213 | 325.384658 | 0 | 0 | 0 | 0.01 | 9613 |
| HTTP | 0.04823423 | 0.21426079 | 0 | 0 | 0 | 0 | 1 |
| HTTPS | 0.05509922 | 0.22817383 | 0 | 0 | 0 | 0 | 1 |
| DNS | 0.00013068 | 0.01143079 | 0 | 0 | 0 | 0 | 1 |
| Telnet | 2.14E-08 | 0.00014635 | 0 | 0 | 0 | 0 | 1 |
| SMTP | 6.43E-08 | 0.00025349 | 0 | 0 | 0 | 0 | 1 |
| SSH | 4.09E-05 | 0.00639772 | 0 | 0 | 0 | 0 | 1 |
| IRC | 1.50E-07 | 0.00038722 | 0 | 0 | 0 | 0 | 1 |
| TCP | 0.57383427 | 0.49451846 | 0 | 0 | 1 | 1 | 1 |
| UDP | 0.21191758 | 0.40866676 | 0 | 0 | 0 | 0 | 1 |
| DHCP | 1.71E-06 | 0.00130903 | 0 | 0 | 0 | 0 | 1 |
| ARP | 6.62E-05 | 0.00813521 | 0 | 0 | 0 | 0 | 1 |
| ICMP | 0.16372157 | 0.37002273 | 0 | 0 | 0 | 0 | 1 |
| IPv | 0.99988731 | 0.01061485 | 0 | 1 | 1 | 1 | 1 |
| LLC | 0.99988731 | 0.01061485 | 0 | 1 | 1 | 1 | 1 |
| Tot sum | 1308.32257 | 2613.30273 | 42 | 525 | 567 | 567.54 | 127335.8 |
| Min | 91.6073456 | 139.695326 | 42 | 50 | 54 | 54 | 13583 |
| Max | 181.963418 | 524.030902 | 42 | 50 | 54 | 55.26 | 49014 |
| AVG | 124.668815 | 240.991485 | 42 | 50 | 54 | 54.0497296 | 13583 |
| Std | 33.3248065 | 160.335722 | 0 | 0 | 0 | 0.37190955 | 12385.2391 |
| Tot size | 124.691567 | 241.549341 | 42 | 50 | 54 | 54.06 | 13583 |
| IAT | 83182525.9 | 17047351.7 | 0 | 83071566 | 83124522.4 | 83343908 | 167639436 |
| Number | 9.49848933 | 0.81915318 | 1 | 9.5 | 9.5 | 9.5 | 15 |
| Magnitue | 13.12182 | 8.62857895 | 9.16515139 | 10 | 10.3923048 | 10.3967148 | 164.821115 |
| Radius | 47.0949848 | 226.769647 | 0 | 0 | 0 | 0.50592128 | 17551.2708 |
| Covariance | 30724.3565 | 323710.68 | 0 | 0 | 0 | 1.34421569 | 154902159 |
| Variance | 0.0964376 | 0.233001 | 0 | 0 | 0 | 0.08 | 1 |
| Weight | 141.51237 | 21.0683073 | 1 | 141.55 | 141.55 | 141.55 | 244.6 |

Với mỗi luồng mạng đều gồm 46 đặc trưng sẽ được gắn 1 nhãn trong số 34 nhãn (33 nhãn cách thức tấn công và 1 nhãn là luồng mạng bình thường). 34 nhãn bao gồm: 

*lưu ý: Nhãn số 11 BenignTraffic có nghĩa là dữ liệu an toàn → truy cập vô hại, không phải tấn công*

*(đây là thống kê từ các file merge)*

| **Số thứ tự** | **Label** | **Số lượng** |
| --- | --- | --- |
| 1 | DDOS-ICMP_FLOOD | 6893259 |
| 2 | DDOS-UDP_FLOOD | 5181027 |
| 3 | DDOS-TCP_FLOOD | 4306086 |
| 4 | DDOS-PSHACK_FLOOD | 3920372 |
| 5 | DDOS-SYN_FLOOD | 3886130 |
| 6 | DDOS-RSTFINFLOOD | 3872808 |
| 7 | DDOS-SYNONYMOUSIP_FLOOD | 3445659 |
| 8 | DOS-UDP_FLOOD | 3177323 |
| 9 | DOS-TCP_FLOOD | 2558256 |
| 10 | DOS-SYN_FLOOD | 1942176 |
| 11 | BENIGN | 1051373 |
| 12 | MIRAI-GREETH_FLOOD | 949381 |
| 13 | MIRAI-UDPPLAIN | 852695 |
| 14 | MIRAI-GREIP_FLOOD | 719655 |
| 15 | DDOS-ICMP_FRAGMENTATION | 433157 |
| 16 | VULNERABILITYSCAN | 357583 |
| 17 | MITM-ARPSPOOFING | 294469 |
| 18 | DDOS-UDP_FRAGMENTATION | 274909 |
| 19 | DDOS-ACK_FRAGMENTATION | 272793 |
| 20 | DNS_SPOOFING | 171468 |
| 21 | RECON-HOSTDISCOVERY | 128677 |
| 22 | RECON-OSSCAN | 93970 |
| 23 | RECON-PORTSCAN | 78730 |
| 24 | DOS-HTTP_FLOOD | 68799 |
| 25 | DDOS-HTTP_FLOOD | 27597 |
| 26 | DDOS-SLOWLORIS | 22400 |
| 27 | DICTIONARYBRUTEFORCE | 12522 |
| 28 | BROWSERHIJACKING | 5630 |
| 29 | COMMANDINJECTION | 5168 |
| 30 | SQLINJECTION | 5022 |
| 31 | XSS | 3705 |
| 32 | BACKDOOR_MALWARE | 3078 |
| 33 | RECON-PINGSWEEP | 2161 |
| 34 | UPLOADING_ATTACK | 1196 |

→ Ta được một bộ dữ liệu với mỗi hàng là thông tin của một luồng mạng gồm 46 đặc trưng và 1 đặc trưng - là label của dữ liệu đó. 

Tổng số lượng mẫu của dataset là: **45019234**

Với lượng dataset này, họ đã chia thành 63 file .csv khác nhau, với mỗi file đã được xáo trộn để có số lượng mẫu thuộc về nhiều label khác nhau → thuận tiện để training.

# 2. Kịch bản

Với bộ dataset này, ta cần phải xử lý dữ liệu nhằm phù hợp với bài toán xây dựng framework lightweight federated learning huấn luyện mô hình IDS trong bối cảnh dữ liệu non-IID.

*Not Independent and Identically Distributed - Không độc lập và không cùng phân phối*

**Kịch bản của tập dữ liệu:**

- Từ data gốc chia ra 20% dùng để test và 80% dùng để train → chia 80% này cho 10 client trong mô hình FL.
- Mỗi client sẽ có số lượng mẫu khác nhau (chia ngẫu nhiên).
- Để giả lập trình trạng non-IID, ta sẽ chia ngẫu nhiên số nhãn, số lượng mẫu trong mỗi nhãn ở từng client.

# 3. Các kỹ thuật sử dụng:

## **Trích chọn đặc trưng:**

Với 46 đặc trưng ban đầu, việc loại bỏ các đặc trưng nhiễu hoặc không mang giá trị phân loại là cần thiết. 

Thay vì loại bỏ thủ công, hệ thống áp dụng logic thống kê để tự động lọc các cột định danh (như cột Number để tránh rò rỉ dữ liệu) và các cột có phương sai cực thấp (giá trị lặp lại chiếm >99%). 

Sau bước lọc sơ bộ, mô hình XGBoost được sử dụng để đánh giá độ quan trọng và trích xuất đúng 25 đặc trưng có giá trị cao nhất thông qua hàm `feature_importances_`. 

Để tối ưu tài nguyên, quá trình tìm đặc trưng này chỉ sử dụng khoảng 10% dữ liệu gốc (tương đương 4,5 triệu dòng), đảm bảo tính đại diện cho toàn bộ tập dữ liệu mà không gây tràn bộ nhớ.

## **Phân chia train/test và chuẩn hóa:**

Toàn bộ dữ liệu với 25 đặc trưng đã chọn và 1 cột nhãn được chia theo tỷ lệ 80/20. Trong đó:

- 20% lượng dữ liệu được giữ lại làm tập test để kiểm thử mô hình và 80% dữ liệu train được dùng để phân bổ cho các client.
- Trước khi phân chia cho client, tập train cần được chuẩn hóa để đưa toàn bộ các giá trị về chung một dải tham chiếu, tránh việc các mô hình cục bộ bị lệch trọng số:
    - Hệ thống sử dụng thuật toán `QuantileTransformer` (để ánh xạ phân phối về dạng chuẩn).
    - Lúc này, thuật toán chỉ gọi hàm `fit_transform()` trên 80% tập train, nghĩa là các tham số thống kê chỉ được tính toán trên dữ liệu lịch sử đã biết.
    - Khi đánh giá trên 20% tập test, hệ thống chỉ gọi hàm `transform()` để ép dữ liệu kiểm thử dùng lại thước đo của tập train.
    
    ⇒ Cơ chế này phản ánh đúng thực tế: một gói tin tấn công mới bay đến sẽ lập tức bị ép vào thang chuẩn hóa của những dữ liệu mạng trong quá khứ mà hệ thống đã từng học, giúp ngăn chặn hoàn toàn rò rỉ dữ liệu.
    

## **Phân bổ dữ liệu non-IID cho 10 client:**

Để tạo ra môi trường dữ liệu không độc lập và không phân phối đồng nhất (non-IID) sát với thực tiễn, thuật toán phân phối Dirichlet với tham số alpha là 0.2 được áp dụng.

Quá trình phân bổ được triển khai qua hai bước ngẫu nhiên: 

- Đầu tiên, hệ thống giới hạn một số lượng nhãn ngẫu nhiên (dao động từ 5 đến 32) cho mỗi client.
- Tiếp theo, với từng nhãn cụ thể, thuật toán Dirichlet sẽ băm tổng số lượng mẫu của nhãn đó thành các phần không đồng đều và gán cho những client đã được cấp phép chứa nhãn đó.

⇒ Kết quả cuối cùng tạo ra sự mất cân bằng sâu sắc giữa 10 client, mỗi bên sẽ sở hữu cấu trúc nhãn và số lượng mẫu hoàn toàn khác biệt.

# 4. Script:

Xử lý bằng Kaggle: 

[fl-prerprocessing-file1.ipynb](fl-prerprocessing-file1.ipynb)

[file2-client-noniid.ipynb](file2-client-noniid.ipynb)

File data toàn cục và file data của từng client đã được chuẩn hóa: 

[](https://drive.google.com/open?id=1-tCCeUgXt_hibyU5Ri_vOOAnG-LhJAxX&usp=drive_fs)

*đã kiểm tra tổng data trên các client bằng đúng 80% tập train*

# 5. Thống kê:

## Các nhãn được giữ lại:

**Nhóm thống kê kích thước gói tin**

- AVG: kích thước trung bình của các gói tin.
- Std: độ lệch chuẩn kích thước gói tin.
- Max: kích thước gói tin lớn nhất.
- Min: kích thước gói tin nhỏ nhất.
- Tot sum: tổng kích thước các gói tin.

**Nhóm thống kê thời gian và tốc độ**

- IAT: khoảng thời gian chờ giữa các gói tin liên tiếp.
- Rate: tốc độ truyền tải gói tin.

**Nhóm trạng thái cờ kết nối**

- syn_flag_number: số lượng cờ đồng bộ.
- rst_flag_number: số lượng cờ đặt lại.
- psh_flag_number: số lượng cờ đẩy dữ liệu.
- ack_flag_number: số lượng cờ xác nhận.
- fin_flag_number: số lượng cờ kết thúc.

**Nhóm đếm tần suất hành vi**

- syn_count: tổng số lần gửi yêu cầu đồng bộ.
- ack_count: tổng số lần phản hồi xác nhận.
- rst_count: tổng số lần yêu cầu đặt lại kết nối.
- fin_count: tổng số lần yêu cầu đóng kết nối.

**Nhóm giao thức tầng mạng và giao vận**

- Protocol Type: mã loại giao thức.
- IPv: giao thức liên mạng cơ bản.
- TCP: giao thức điều khiển truyền vận.
- UDP: giao thức truyền tải không kết nối.
- ICMP: giao thức bản tin điều khiển mạng.

**Nhóm giao thức phân giải**

- DNS: giao thức phân giải tên miền.
- ARP: giao thức phân giải địa chỉ.

**Nhóm cấu trúc gói tin**

- Header_Length: độ dài phần đầu của gói tin.
- Time_To_Live: giới hạn thời gian tồn tại của gói tin.

## Thống kê data ở các client:

**client 1 (tổng: 31.520 mẫu)**

- RECON-OSSCAN: 27.656
- MIRAI-GREIP_FLOOD: 2.192
- DOS-UDP_FLOOD: 983
    - BROWSERHIJACKING: 413
- DICTIONARYBRUTEFORCE: 125
- DDOS-ACK_FRAGMENTATION: 95
- RECON-HOSTDISCOVERY: 54
- VULNERABILITYSCAN: 2

**client 2 (tổng: 2.211.589 mẫu)**

- DDOS-PSHACK_FLOOD: 532.342
- DDOS-UDP_FLOOD: 519.111
- DDOS-SYN_FLOOD: 382.191
- MIRAI-UDPPLAIN: 274.824
- DDOS-UDP_FRAGMENTATION: 156.454
- DDOS-RSTFINFLOOD: 147.137
- DDOS-ICMP_FLOOD: 55.369
- MITM-ARPSPOOFING: 26.804
- DDOS-SYNONYMOUSIP_FLOOD: 23.009
- DDOS-ACK_FRAGMENTATION: 20.031
- RECON-HOSTDISCOVERY: 19.269
- DDOS-SLOWLORIS: 14.449
- DOS-HTTP_FLOOD: 11.654
- VULNERABILITYSCAN: 11.106
- BENIGN: 10.608
- DICTIONARYBRUTEFORCE: 5.200
- RECON-PINGSWEEP: 853
- MIRAI-GREETH_FLOOD: 709
- UPLOADING_ATTACK: 263
- DOS-SYN_FLOOD: 130
- XSS: 76

**client 3 (tổng: 3.728.450 mẫu)**

- DDOS-UDP_FLOOD: 1.774.100
- DOS-SYN_FLOOD: 1.450.758
- DDOS-ACK_FRAGMENTATION: 165.304
- VULNERABILITYSCAN: 135.220
- BENIGN: 54.689
- MIRAI-GREETH_FLOOD: 44.098
- DDOS-ICMP_FRAGMENTATION: 26.963
- DNS_SPOOFING: 18.172
- DOS-HTTP_FLOOD: 15.625
- MIRAI-GREIP_FLOOD: 15.112
- DDOS-HTTP_FLOOD: 11.669
- RECON-HOSTDISCOVERY: 6.229
- BROWSERHIJACKING: 3.319
- COMMANDINJECTION: 2.902
- BACKDOOR_MALWARE: 1.501
- XSS: 1.441
- RECON-PINGSWEEP: 752
- SQLINJECTION: 219
- UPLOADING_ATTACK: 178
- DICTIONARYBRUTEFORCE: 119
- DDOS-UDP_FRAGMENTATION: 50
- MIRAI-UDPPLAIN: 17
- RECON-OSSCAN: 7
- DDOS-SLOWLORIS: 3
- MITM-ARPSPOOFING: 3

**client 4 (tổng: 1.920.254 mẫu)**

- DDOS-PSHACK_FLOOD: 1.594.735
- MITM-ARPSPOOFING: 198.184
- MIRAI-GREIP_FLOOD: 106.803
- RECON-PORTSCAN: 11.575
- DOS-HTTP_FLOOD: 4.134
- MIRAI-GREETH_FLOOD: 3.353
- DDOS-HTTP_FLOOD: 1.469
- RECON-HOSTDISCOVERY: 1

**client 5 (tổng: 2.531.881 mẫu)**

- DDOS-UDP_FLOOD: 1.850.447
- DDOS-RSTFINFLOOD: 616.498
- DNS_SPOOFING: 17.843
- DDOS-ICMP_FLOOD: 11.299
- RECON-PORTSCAN: 11.150
- MITM-ARPSPOOFING: 10.568
- VULNERABILITYSCAN: 6.752
- DDOS-UDP_FRAGMENTATION: 3.646
- XSS: 1.443
- BROWSERHIJACKING: 723
- DDOS-ACK_FRAGMENTATION: 653
- BACKDOOR_MALWARE: 461
- MIRAI-UDPPLAIN: 366
- UPLOADING_ATTACK: 32

**client 6 (tổng: 2.686.617 mẫu)**

- DDOS-RSTFINFLOOD: 2.274.996
- DDOS-ICMP_FLOOD: 406.912
- MIRAI-GREIP_FLOOD: 2.864
- DDOS-SLOWLORIS: 1.615
- DDOS-ACK_FRAGMENTATION: 230

**client 7 (tổng: 1.824.975 mẫu)**

- DDOS-SYNONYMOUSIP_FLOOD: 1.577.672
- DDOS-TCP_FLOOD: 143.819
- DNS_SPOOFING: 63.786
- DOS-HTTP_FLOOD: 14.087
- DDOS-ICMP_FRAGMENTATION: 12.005
- DDOS-PSHACK_FLOOD: 7.061
- DDOS-HTTP_FLOOD: 3.491
- RECON-HOSTDISCOVERY: 1.906
- DOS-UDP_FLOOD: 1.096
- RECON-PINGSWEEP: 50
- BROWSERHIJACKING: 2

**client 8 (tổng: 12.759.509 mẫu)**

- DDOS-ICMP_FLOOD: 3.116.185
- DDOS-SYN_FLOOD: 2.726.649
- DOS-UDP_FLOOD: 2.133.918
- DOS-TCP_FLOOD: 2.046.555
- DDOS-SYNONYMOUSIP_FLOOD: 1.071.571
- BENIGN: 775.753
- MIRAI-UDPPLAIN: 406.890
- VULNERABILITYSCAN: 132.981
- DOS-SYN_FLOOD: 89.079
- DDOS-RSTFINFLOOD: 59.592
- RECON-HOSTDISCOVERY: 51.041
- DDOS-PSHACK_FLOOD: 48.451
- DDOS-ACK_FRAGMENTATION: 31.899
- RECON-PORTSCAN: 23.335
- RECON-OSSCAN: 12.700
- DNS_SPOOFING: 11.975
- DOS-HTTP_FLOOD: 9.536
- DDOS-HTTP_FLOOD: 5.449
- DDOS-SLOWLORIS: 1.283
- DICTIONARYBRUTEFORCE: 1.242
- COMMANDINJECTION: 1.229
- DDOS-UDP_FLOOD: 1.117
- BACKDOOR_MALWARE: 498
- UPLOADING_ATTACK: 482
- RECON-PINGSWEEP: 70
- MIRAI-GREETH_FLOOD: 24
- DDOS-ICMP_FRAGMENTATION: 3
- MITM-ARPSPOOFING: 2

**client 9 (tổng: 4.354.192 mẫu)**

- DDOS-TCP_FLOOD: 3.300.973
- MIRAI-GREIP_FLOOD: 448.710
- DDOS-ICMP_FLOOD: 391.890
- DDOS-SYNONYMOUSIP_FLOOD: 84.249
- RECON-HOSTDISCOVERY: 24.436
- DNS_SPOOFING: 23.748
- DDOS-UDP_FRAGMENTATION: 23.574
- RECON-PORTSCAN: 16.923
- DOS-SYN_FLOOD: 13.740
- DDOS-PSHACK_FLOOD: 13.670
- DOS-UDP_FLOOD: 6.535
- DICTIONARYBRUTEFORCE: 3.332
- SQLINJECTION: 2.292
- MIRAI-GREETH_FLOOD: 85
- DDOS-SLOWLORIS: 20
- MIRAI-UDPPLAIN: 5
- RECON-PINGSWEEP: 4
- XSS: 4
- UPLOADING_ATTACK: 2

**client 10 (tổng: 3.965.607 mẫu)**

- DDOS-ICMP_FLOOD: 1.532.902
- DDOS-PSHACK_FLOOD: 940.006
- MIRAI-GREETH_FLOOD: 711.189
- DOS-UDP_FLOOD: 399.242
- DDOS-ICMP_FRAGMENTATION: 307.521
- DDOS-UDP_FRAGMENTATION: 36.179
- RECON-OSSCAN: 34.810
- DNS_SPOOFING: 1.646
- SQLINJECTION: 1.506
- DDOS-SLOWLORIS: 549
- BROWSERHIJACKING: 47
- RECON-HOSTDISCOVERY: 5
- COMMANDINJECTION: 3
- BACKDOOR_MALWARE: 2