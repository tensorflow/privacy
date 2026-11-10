load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_binary(
    name = "example",
    srcs = ["example.py"],
    type_checking = True,
    deps = [
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack",
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack:privacy_report",
    ],
)
