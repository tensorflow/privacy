package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "dp_keras_model",
    srcs = [
        "dp_keras_model.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/compiler/jit:xla_cpu_jit",
        "//third_party/tensorflow/compiler/jit:xla_gpu_jit",
    ],
)
