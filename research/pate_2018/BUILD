load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "ignore_srcs",
    srcs = [
        "core.py",
        "core_test.py",
        "smooth_sensitivity.py",
        "smooth_sensitivity_test.py",
    ],
    tags = ["ignore_srcs"],
)

py_library(
    name = "core",
    srcs = ["core.py"],
)
