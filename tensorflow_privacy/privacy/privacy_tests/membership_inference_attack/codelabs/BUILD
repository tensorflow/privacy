load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "example",
    srcs = ["example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack",
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack:privacy_report",
    ],
)
