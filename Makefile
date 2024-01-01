demo:	demo.o custom
	nvcc -o demo -lm -lcuda -lrt demo.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o src/optimizer/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

demo.o:	demo.cc
	nvcc --compile demo.cc -I./ -L/usr/local/cuda/lib64 -lcudart

train_model:	demo
	./demo

main: main.o dnnNetwork.o network.o mnist.o layer loss optimizer
	nvcc -o main -lm -lcuda -lrt main.o dnnNetwork.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

main.o: main.cc
	nvcc --compile main.cc -o main.o -I./ -L/usr/local/cuda/lib64 -lcudart

dnnNetwork.o: dnnNetwork.cc
	nvcc --compile dnnNetwork.cc -o dnnNetwork.o -I./ -L/usr/local/cuda/lib64 -lcudart

network.o: src/network.cc
	nvcc --compile src/network.cc -o src/network.o -I./ -L/usr/local/cuda/lib64 -lcudart

mnist.o: src/mnist.cc
	nvcc --compile src/mnist.cc -o src/mnist.o  -I./ -L/usr/local/cuda/lib64 -lcudart

layer: src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
	nvcc --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/conv.cc -o src/layer/conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/CPU_conv.cc -o src/layer/CPU_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/GPU_conv.cc -o src/layer/GPU_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/relu.cc -o src/layer/relu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -L/usr/local/cuda/lib64 -lcudart

gpu_basic:
	rm -f src/layer/*.o
	nvcc --compile src/layer/new_layer/CPU_forward_conv.cc -o src/layer/CPU_forward_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_forward_conv.cu -o src/layer/GPU_forward_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_utils.cu -o src/layer/GPU_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_new_forward_basic.cu -o src/layer/GPU_new_forward_basic.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/My_GPU.cu -o src/layer/My_GPU.o -I./ -L/usr/local/cuda/lib64 -lcudart 

gpu_opver1:
	rm -f src/layer/*.o
	nvcc --compile src/layer/new_layer/CPU_forward_conv.cc -o src/layer/CPU_forward_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_forward_conv.cu -o src/layer/GPU_forward_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_utils.cu -o src/layer/GPU_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_new_forward_basic.cu -o src/layer/GPU_new_forward_basic.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/My_GPU_Ver1.cu -o src/layer/My_GPU_Ver1.o -I./ -L/usr/local/cuda/lib64 -lcudart 

gpu_opver2:
	rm -f src/layer/*.o
	nvcc --compile src/layer/new_layer/CPU_forward_conv.cc -o src/layer/CPU_forward_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_forward_conv.cu -o src/layer/GPU_forward_conv.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_utils.cu -o src/layer/GPU_utils.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/GPU_new_forward_basic.cu -o src/layer/GPU_new_forward_basic.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/new_layer/My_GPU_Ver2.cu -o src/layer/My_GPU_Ver2.o -I./ -L/usr/local/cuda/lib64 -lcudart 

	

loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
	nvcc --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart

optimizer: src/optimizer/sgd.cc
	nvcc --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -L/usr/local/cuda/lib64 -lcudart

setup: 
#	make dnnNetwork.o
#	make network.o
#	make mnist.o
#	make layer
#	make loss
#	make optimizer
	make main

clean:
##	rm -f infoGPU infoGPU.o main main.o
	rm main
	rm main.o
run: main
	./main