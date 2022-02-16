load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "secret_sharer",
    srcs = ["__init__.py"],
)

py_library(
    name = "exposures",
    srcs = ["exposures.py"],
    srcs_version = "PY3",
)

py_test(
    name = "exposures_test",
    srcs = ["exposures_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":exposures"],
)

py_library(
    name = "generate_secrets",
    srcs = ["generate_secrets.py"],
    srcs_version = "PY3",
)

py_test(
    name = "generate_secrets_test",
    srcs = ["generate_secrets_test.py"],
    deps = [":generate_secrets"],
)
