load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "tensorflow_privacy_v1",
    srcs = ["__init__.py"],
    deps = [
        "//tensorflow_privacy/privacy/estimators/v1:dnn",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_vectorized",
    ],
)

build_test(
    name = "tensorflow_privacy_build_test",
    targets = [":tensorflow_privacy_v1"],
)
