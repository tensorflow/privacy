load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
    srcs_version = "PY3",
)

py_library(
    name = "dnn",
    srcs = [
        "dnn.py",
    ],
    srcs_version = "PY3",
    deps = [":head"],
)

py_test(
    name = "head_test",
    timeout = "long",
    srcs = ["head_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dnn",
        "//tensorflow_privacy/privacy/estimators:test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)
