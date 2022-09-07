load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "logistic_regression",
    srcs = ["__init__.py"],
)

py_library(
    name = "multinomial_logistic",
    srcs = ["multinomial_logistic.py"],
    srcs_version = "PY3",
    deps = [
        ":datasets",
        ":single_layer_softmax",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "multinomial_logistic_test",
    size = "large",
    srcs = ["multinomial_logistic_test.py"],
    local = True,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":datasets",
        ":multinomial_logistic",
        "//tensorflow_privacy/privacy/analysis:compute_dp_sgd_privacy_lib",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    srcs_version = "PY3",
)

py_test(
    name = "datasets_test",
    size = "small",
    srcs = ["datasets_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-net:external"],
    deps = [":datasets"],
)

py_library(
    name = "single_layer_softmax",
    srcs = ["single_layer_softmax.py"],
    srcs_version = "PY3",
    deps = [":datasets"],
)

py_test(
    name = "single_layer_softmax_test",
    size = "medium",
    srcs = ["single_layer_softmax_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":datasets",
        ":single_layer_softmax",
    ],
)
