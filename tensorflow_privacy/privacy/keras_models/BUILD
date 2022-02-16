load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "keras_models",
    srcs = ["__init__.py"],
)

py_library(
    name = "dp_keras_model",
    srcs = [
        "dp_keras_model.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "dp_keras_model_test",
    srcs = ["dp_keras_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/keras_models:dp_keras_model"],
)
