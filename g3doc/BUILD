load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy",
        "//third_party/py/tensorflow_docs/api_generator:doc_controls",
        "//third_party/py/tensorflow_docs/api_generator:generate_lib",
        "//third_party/py/tensorflow_docs/api_generator:public_api",
    ],
)
