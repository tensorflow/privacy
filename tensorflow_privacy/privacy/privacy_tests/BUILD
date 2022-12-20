load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "privacy_tests",
    srcs = ["__init__.py"],
)

py_test(
    name = "utils_test",
    timeout = "long",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils"],
)

py_test(
    name = "epsilon_lower_bound_test",
    srcs = ["epsilon_lower_bound_test.py"],
    deps = [":epsilon_lower_bound"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "epsilon_lower_bound",
    srcs = ["epsilon_lower_bound.py"],
)
