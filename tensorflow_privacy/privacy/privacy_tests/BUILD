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
    deps = [
        ":utils",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # google-only
    ],
)

py_test(
    name = "epsilon_lower_bound_test",
    srcs = ["epsilon_lower_bound_test.py"],
    deps = [
        ":epsilon_lower_bound",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # google-only
    ],
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
