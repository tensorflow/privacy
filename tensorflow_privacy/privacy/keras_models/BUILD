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

py_test(
    name = "dp_keras_model_test",
    srcs = ["dp_keras_model_test.py"],
    python_version = "PY3",
    deps = [
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_privacy/privacy/keras_models:dp_keras_model",
    ],
)
