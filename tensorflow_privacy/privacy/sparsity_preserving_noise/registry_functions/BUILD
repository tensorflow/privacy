load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "embedding",
    srcs = ["embedding.py"],
    type_checking = True,
    deps = ["//tensorflow_privacy/privacy/sparsity_preserving_noise:type_aliases"],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    deps = [":embedding"],
)
