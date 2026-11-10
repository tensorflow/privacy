load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "privacy_tests",
    srcs = ["__init__.py"],
    type_checking = True,
    deps = [
        ":version",
    ],
)

py_test(
    name = "utils_test",
    timeout = "long",
    srcs = ["utils_test.py"],
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
    type_checking = True,
)

py_library(
    name = "epsilon_lower_bound",
    srcs = ["epsilon_lower_bound.py"],
    type_checking = True,
)

py_library(
    name = "version",
    srcs = ["version.py"],
    type_checking = True,
)
