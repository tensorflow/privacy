package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "v1",
    srcs = ["__init__.py"],
)

py_library(
    name = "head",
    srcs = [
        "head.py",
    ],
)

py_library(
    name = "dnn",
    srcs = [
        "dnn.py",
    ],
    deps = [":head"],
)

py_library(
    name = "linear",
    srcs = [
        "linear.py",
    ],
    deps = [":head"],
)

py_test(
    name = "head_test",
    timeout = "long",
    srcs = ["head_test.py"],
    deps = [
        ":head",
        "//tensorflow_privacy/privacy/estimators:test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)

py_test(
    name = "dnn_test",
    timeout = "long",
    srcs = ["dnn_test.py"],
    deps = [
        ":dnn",
        "//tensorflow_privacy/privacy/estimators:test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)

py_test(
    name = "linear_test",
    timeout = "long",
    srcs = ["linear_test.py"],
    deps = [
        ":linear",
        "//tensorflow_privacy/privacy/estimators:test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)
