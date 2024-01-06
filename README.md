# Parallel_CNN_network
Trong dự án cuối kì, nhóm sẽ triển khai và tối ưu hóa quá trình chuyển tiếp của convolutional layer bằng CUDA. Các convolutional layer là các khối xây dựng chính của convolutional neural network (CNN), được sử dụng trong nhiều tác vụ học máy như phân loại hình ảnh, phát hiện đối tượng, xử lý ngôn ngữ tự nhiên và hệ thống đề xuất. Nhìn chung, CNN hoạt động tốt với các nhiệm vụ trong đó các tính năng dữ liệu đầu vào có một mức độ quan hệ không gian nào đó.

Triển khai trên CUDA, nhóm sẽ tiến hành cài đặt tối ưu hóa convolutional layer để thực hiện lan truyền tới cho các lớp **C1** và **C3**

- Sử dụng open source [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) (Mini-DNN) framework để thực hiện cài đặt phiên bản LetNet-5:
 - Lớp convolution C1:
   * Đầu vào: vector có kích thước 28x28
   * Đầu ra: 6 lớp có kích thước 14x14
 -	Lớp convolution C3:
   * Đầu vào: 6 lớp có kích thước 1x12
    * Đầu ra: 16 lớp có kích thước 10x10


- Sử dụng bộ dữ liệu [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) gồm các hình ảnh (có kích thước 28x28 pixel)



> Động lực thực hiện: Để có thêm hiểu biết về lập trình trong song được ứng dụng vào thực tế trong máy học, giúp tối ưu hóa thời gian của việc huấn luyện của mô hình. Để có cái nhìn tổng quát hơn về các ứng dụng của lập trình song song vào các vấn đề khác.




Cuối cùng, đồ án này mục đích sử dụng CUDA và các phương pháp tối ưu hóa bằng cách thiết kế và triển khai theo chiều xuôi của **Neural-Network Convolutional** trên **Colab**.

## Bộ dữ liệu
Tải về và giải nén [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset in `./data/mnist/`.

## Train model
```shell
!make gpu_basic
!make setup
!make train_model
```

Run `./demo`.

Result: 
simple neural network with 3 FC layers can obtain `0.97+` accuracy on MNIST testset.
LeNet-like CNN can obtain `0.98+` accuracy on MNIST testset.
