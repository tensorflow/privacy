load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

licenses(["notice"])

py_library(
    name = "mnist_dpsgd_tutorial_common",
    srcs = ["mnist_dpsgd_tutorial_common.py"],
    type_checking = True,
)

py_binary(
    name = "mnist_dpsgd_tutorial",
    srcs = ["mnist_dpsgd_tutorial.py"],
    type_checking = True,
    deps = [
        ":mnist_dpsgd_tutorial_common",
        "//tensorflow_privacy/privacy/analysis:compute_dp_sgd_privacy_lib",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)

py_binary(
    name = "mnist_dpsgd_tutorial_eager",
    srcs = ["mnist_dpsgd_tutorial_eager.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/optimizers:dp_optimizer"],
)

py_binary(
    name = "mnist_dpsgd_tutorial_keras",
    srcs = ["mnist_dpsgd_tutorial_keras.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras"],
)

py_binary(
    name = "mnist_dpsgd_tutorial_keras_model",
    srcs = ["mnist_dpsgd_tutorial_keras_model.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/keras_models:dp_keras_model"],
)

py_binary(
    name = "mnist_dpsgd_tutorial_vectorized",
    srcs = ["mnist_dpsgd_tutorial_vectorized.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/optimizers:dp_optimizer_vectorized"],
)

py_binary(
    name = "mnist_dpsgd_tutorial_tpu",
    srcs = ["mnist_dpsgd_tutorial_tpu.py"],
    type_checking = True,
    deps = [
        ":mnist_dpsgd_tutorial_common",
        "//tensorflow_privacy/privacy/analysis:compute_dp_sgd_privacy_lib",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)

py_binary(
    name = "mnist_lr_tutorial",
    srcs = ["mnist_lr_tutorial.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/optimizers:dp_optimizer"],
)

py_binary(
    name = "lm_dpsgd_tutorial",
    srcs = ["lm_dpsgd_tutorial.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/optimizers:dp_optimizer"],
)

py_binary(
    name = "movielens_tutorial",
    srcs = ["movielens_tutorial.py"],
    type_checking = True,
    deps = [
        "//tensorflow_privacy/privacy/analysis:gdp_accountant",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)

filegroup(
    name = "ignore_srcs",
    srcs = ["bolton_tutorial.py"],
    tags = ["ignore_srcs"],
)
