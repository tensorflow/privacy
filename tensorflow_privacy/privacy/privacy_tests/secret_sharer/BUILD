load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "secret_sharer",
    srcs = ["__init__.py"],
    type_checking = True,
)

py_library(
    name = "exposures",
    srcs = ["exposures.py"],
    type_checking = True,
)

py_test(
    name = "exposures_test",
    srcs = ["exposures_test.py"],
    deps = [":exposures"],
)

py_library(
    name = "generate_secrets",
    srcs = ["generate_secrets.py"],
    type_checking = True,
)

py_test(
    name = "generate_secrets_test",
    srcs = ["generate_secrets_test.py"],
    deps = [":generate_secrets"],
)
