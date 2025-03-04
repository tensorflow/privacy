package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "logistic_regression",
    srcs = ["__init__.py"],
)

py_library(
    name = "multinomial_logistic",
    srcs = ["multinomial_logistic.py"],
    deps = [
        ":datasets",
        ":single_layer_softmax",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "multinomial_logistic_test",
    size = "large",
    srcs = ["multinomial_logistic_test.py"],
    local = True,
    deps = [
        ":datasets",
        ":multinomial_logistic",
        "//tensorflow_privacy/privacy/analysis:compute_dp_sgd_privacy_lib",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    deps = ["//third_party/py/sklearn:v1_0_2"],
)

py_test(
    name = "datasets_test",
    size = "small",
    srcs = ["datasets_test.py"],
    tags = ["requires-net:external"],
    deps = [":datasets"],
)

py_library(
    name = "single_layer_softmax",
    srcs = ["single_layer_softmax.py"],
    deps = [":datasets"],
)

py_test(
    name = "single_layer_softmax_test",
    size = "medium",
    srcs = ["single_layer_softmax_test.py"],
    deps = [
        ":datasets",
        ":single_layer_softmax",
    ],
)
