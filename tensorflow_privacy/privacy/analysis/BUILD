load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "analysis",
    srcs = ["__init__.py"],
)

py_library(
    name = "compute_dp_sgd_privacy_lib",
    srcs = ["compute_dp_sgd_privacy_lib.py"],
    srcs_version = "PY3",
    deps = ["@com_google_differential_py//dp_accounting:accounting"],
)

py_binary(
    name = "compute_dp_sgd_privacy",
    srcs = ["compute_dp_sgd_privacy.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":compute_dp_sgd_privacy_lib"],
)

py_test(
    name = "compute_dp_sgd_privacy_test",
    size = "small",
    srcs = ["compute_dp_sgd_privacy_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":compute_dp_sgd_privacy_lib"],
)

py_binary(
    name = "compute_noise_from_budget",
    srcs = ["compute_noise_from_budget.py"],
    deps = [":compute_noise_from_budget_lib"],
)

py_library(
    name = "compute_noise_from_budget_lib",
    srcs = ["compute_noise_from_budget_lib.py"],
    deps = ["@com_google_differential_py//dp_accounting:accounting"],
)

py_test(
    name = "compute_noise_from_budget_test",
    srcs = ["compute_noise_from_budget_test.py"],
    deps = [":compute_noise_from_budget_lib"],
)

py_library(
    name = "gdp_accountant",
    srcs = ["gdp_accountant.py"],
)

py_library(
    name = "tensor_buffer",
    srcs = ["tensor_buffer.py"],
    srcs_version = "PY3",
)

py_test(
    name = "tensor_buffer_eager_test",
    size = "small",
    srcs = ["tensor_buffer_eager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tensor_buffer"],
)

py_test(
    name = "tensor_buffer_graph_test",
    size = "small",
    srcs = ["tensor_buffer_graph_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tensor_buffer"],
)

py_library(
    name = "tree_aggregation_accountant",
    srcs = ["tree_aggregation_accountant.py"],
    srcs_version = "PY3",
)

py_test(
    name = "tree_aggregation_accountant_test",
    srcs = ["tree_aggregation_accountant_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tree_aggregation_accountant",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)
