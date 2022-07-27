load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "LICENSE",
])

py_library(
    name = "tensorflow_privacy",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":version",
        "//tensorflow_privacy/privacy/analysis:compute_dp_sgd_privacy_lib",
        "//tensorflow_privacy/privacy/analysis:tree_aggregation_accountant",
        "//tensorflow_privacy/privacy/dp_query",
        "//tensorflow_privacy/privacy/dp_query:discrete_gaussian_query",
        "//tensorflow_privacy/privacy/dp_query:distributed_discrete_gaussian_query",
        "//tensorflow_privacy/privacy/dp_query:distributed_skellam_query",
        "//tensorflow_privacy/privacy/dp_query:gaussian_query",
        "//tensorflow_privacy/privacy/dp_query:nested_query",
        "//tensorflow_privacy/privacy/dp_query:no_privacy_query",
        "//tensorflow_privacy/privacy/dp_query:normalized_query",
        "//tensorflow_privacy/privacy/dp_query:quantile_adaptive_clip_sum_query",
        "//tensorflow_privacy/privacy/dp_query:quantile_estimator_query",
        "//tensorflow_privacy/privacy/dp_query:restart_query",
        "//tensorflow_privacy/privacy/dp_query:tree_aggregation",
        "//tensorflow_privacy/privacy/dp_query:tree_aggregation_query",
        "//tensorflow_privacy/privacy/dp_query:tree_range_query",
        "//tensorflow_privacy/privacy/estimators:dnn",
        "//tensorflow_privacy/privacy/keras_models:dp_keras_model",
        "//tensorflow_privacy/privacy/logistic_regression:datasets",
        "//tensorflow_privacy/privacy/logistic_regression:multinomial_logistic",
        "//tensorflow_privacy/privacy/logistic_regression:single_layer_softmax",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras_vectorized",
        "//tensorflow_privacy/v1:tensorflow_privacy_v1",
    ],
)

build_test(
    name = "tensorflow_privacy_build_test",
    targets = [":tensorflow_privacy"],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY3",
)

filegroup(
    name = "ignore_srcs",
    tags = ["ignore_srcs"],
)
