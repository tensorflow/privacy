load("@rules_python//python:defs.bzl", "py_binary")

licenses(["notice"])

py_binary(
    name = "mnist_scratch",
    srcs = ["mnist_scratch.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow:tensorflow_compat_v1_estimator",
        "//third_party/py/tensorflow:tensorflow_estimator",
    ],
)
