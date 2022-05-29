bench: main.cpp
	g++ -o bench -O3 main.cpp -Ionnxruntime/onnxruntime-1.11.1/include/onnxruntime/core/session -Ionnxruntime/onnxruntime-1.11.1/include -Lonnxruntime/onnxruntime-osx-universal2-1.11.1/lib -lonnxruntime --std=c++11
