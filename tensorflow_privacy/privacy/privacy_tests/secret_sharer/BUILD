package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "secret_sharer",
    srcs = ["__init__.py"],
)

py_library(
    name = "exposures",
    srcs = ["exposures.py"],
)

py_test(
    name = "exposures_test",
    srcs = ["exposures_test.py"],
    deps = [":exposures"],
)

py_library(
    name = "generate_secrets",
    srcs = ["generate_secrets.py"],
)

py_test(
    name = "generate_secrets_test",
    srcs = ["generate_secrets_test.py"],
    deps = [":generate_secrets"],
)
