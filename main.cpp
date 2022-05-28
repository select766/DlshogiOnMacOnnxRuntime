#include <iostream>
#include <fstream>
#include <chrono>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/providers/coreml/coreml_provider_factory.h>

using namespace std;

const int test_case_size = 1024;
const int input_1_ch = 62, input_2_ch = 57;
const int input_spatial = 9 * 9;
const int policy_size = 2187;
const int value_size = 1;

void read_test_case(vector<float> &input_1, vector<float> &input_2, vector<float> &output_policy, vector<float> &output_value) {
    ifstream fin("SampleIO15x224MyData.bin", ios::in | ios::binary);
    if (!fin) {
        cerr << "failed to open test case" << endl;
        exit(1);
    }

    // input: [test_case_size, 119, 9, 9], output_policy: [test_case_size, 2187], output_value: [test_case_size, 1]
    // 全てfloat32でこの順で保存されている
    // output_valueはsigmoidがかかっていない
    // ONNXモデルではinput_1: [batch_size, 62, 9, 9], input_2: [batch_size, 57, 9, 9]に分ける必要がある

    input_1.resize(test_case_size * input_1_ch * input_spatial * sizeof(float));
    input_2.resize(test_case_size * input_2_ch * input_spatial * sizeof(float));
    output_policy.resize(test_case_size * policy_size * sizeof(float));
    output_value.resize(test_case_size * value_size * sizeof(float));
    
    for (int i = 0; i < test_case_size; i++) {
        fin.read((char*)&(input_1[i * input_1_ch * input_spatial]), input_1_ch * input_spatial * sizeof(float));
        fin.read((char*)&(input_2[i * input_2_ch * input_spatial]), input_2_ch * input_spatial * sizeof(float));
    }
    fin.read((char*)&(output_policy[0]), test_case_size * policy_size * sizeof(float));
    fin.read((char*)&(output_value[0]), test_case_size * value_size * sizeof(float));
    if (!fin) {
        cerr << "failed to read test case" << endl;
        exit(1);
    }
}

void check_result(const float* expected, const float* actual, int count, float rtol = 1e-1, float atol = 5e-2) {
    float max_diff = 0.0F;
    for (int i = 0; i < count; i++) {
        auto e = expected[i];
        auto a = actual[i];
        auto diff = abs(e - a);
        auto tol = atol + rtol * abs(e);
        if (diff > tol) {
            cerr << "Error at index " << i << ": " << e << " != " << a << endl;
            return;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    cerr << "Max difference: " << max_diff << endl;
}

int main(int argc, const char** argv) {
    if (argc != 4) {
        cerr << "bench batch_size backend run_time_sec" << endl;
        cerr << "backend: cpu, coreml" << endl;
        cerr << "example: ./bench 16 coreml 60" << endl;
        exit(1);
    }
    const int batch_size = atoi(argv[1]);
    const double run_time = atof(argv[3]);
    const char* backend = argv[2];
    vector<float> input_1, input_2, output_policy, output_value, output_policy_expected, output_value_expected;
    output_policy.resize(test_case_size * policy_size * sizeof(float));
    output_value.resize(test_case_size * value_size * sizeof(float));
    cerr << "reading test case" << endl;
    read_test_case(input_1, input_2, output_policy_expected, output_value_expected);
    cerr << "reading model" << endl;

    Ort::SessionOptions session_options;
    session_options.DisableMemPattern();
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
    if (strcmp(backend, "cpu") == 0) {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, true));
    } else if(strcmp(backend, "coreml") == 0) {
        uint32_t coreml_flags = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
    } else {
        cerr << "Unknown backend " << backend << endl;
        exit(1);
    }
    // std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Session session_{env, "model_resnet15x224_swish-072.onnx", session_options};
    std::array<int64_t, 4> input_1_shape{batch_size, input_1_ch, 9, 9};
    std::array<int64_t, 4> input_2_shape{batch_size, input_2_ch, 9, 9};
    std::array<int64_t, 2> output_policy_shape{batch_size, policy_size};
    std::array<int64_t, 2> output_value_shape{batch_size, value_size};
    const char* input_names[] = {"input1", "input2"};
    const char* output_names[] = {"output_policy", "output_value"};
    std::array<Ort::Value, 2> input_values{
        Ort::Value::CreateTensor<float>(memory_info, input_1.data(), batch_size * input_1_ch * input_spatial, input_1_shape.data(), input_1_shape.size()),
        Ort::Value::CreateTensor<float>(memory_info, input_2.data(), batch_size * input_2_ch * input_spatial, input_2_shape.data(), input_2_shape.size())
    };
    std::array<Ort::Value, 2> output_values{
        Ort::Value::CreateTensor<float>(memory_info, output_policy.data(), batch_size * policy_size, output_policy_shape.data(), output_policy_shape.size()),
        Ort::Value::CreateTensor<float>(memory_info, output_value.data(), batch_size * value_size, output_value_shape.data(), output_value_shape.size())
    };

    // run first time
    session_.Run(Ort::RunOptions{nullptr}, input_names, input_values.data(), input_values.size(), output_names, output_values.data(), output_values.size());
    cerr << "Comparing output_policy to test case" << endl;
    check_result(output_policy_expected.data(), output_policy.data(), batch_size * policy_size);
    cerr << "Comparing output_value to test case" << endl;
    check_result(output_value_expected.data(), output_value.data(), batch_size * value_size);

    cerr << "Benchmarking for " << run_time << " sec..." << endl;
    auto start_time = chrono::system_clock::now();
    chrono::duration<int, micro> elapsed;
    int run_count = 0;
    // run in loop
    while (1) {
        elapsed = chrono::system_clock::now() - start_time;
        if (elapsed.count() > run_time * 1000000.0) {
            break;
        }
        session_.Run(Ort::RunOptions{nullptr}, input_names, input_values.data(), input_values.size(), output_names, output_values.data(), output_values.size());
        run_count++;
    }

    double elapsed_sec = double(elapsed.count()) / 1000000.0;
    double inference_per_sec = double(run_count) / elapsed_sec;
    double sample_per_sec = inference_per_sec * batch_size;
    cout << "Backend: " << backend << ", batch size: " << batch_size << endl;
    cout << "Run for " << elapsed_sec << " sec" << endl << inference_per_sec << " inference / sec" << endl << sample_per_sec << " samples / sec" << endl;

    return 0;
}
