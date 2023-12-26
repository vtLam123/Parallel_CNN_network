# demo: demo.o custom
# 	nvcc -o demo -lm -lcuda -lrt demo.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o src/optimizer/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

# demo.o: demo.cc
# 	nvcc --compile demo.cc -I./ -L/usr/local/cuda/lib64 -lcudart

############################################################################

main: main.o dnnNetwork.o network.o mnist.o layer new_layer loss optimizer
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
	nvcc --compile src/layer/conv-cpu.cc -o src/layer/conv-cpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/conv-gpu.cc -o src/layer/conv-gpu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/relu.cc -o src/layer/relu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -L/usr/local/cuda/lib64 -lcudart

custom:
	nvcc --compile src/layer/custom/cpu-new-forward.cc -o src/layer/custom/cpu-new-forward.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/custom/gpu-utils.cu -o src/layer/custom/gpu-utils.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/layer/custom/gpu-new-forward-basic.cu -o src/layer/custom/new-forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
	nvcc --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart

optimizer: src/optimizer/sgd.cc
	nvcc --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f infoGPU infoGPU.o main main.o

run: main
	./main