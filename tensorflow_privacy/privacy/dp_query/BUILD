load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "__init__",  # There is already a dp_query.
    srcs = ["__init__.py"],
)

py_library(
    name = "dp_query",
    srcs = ["dp_query.py"],
    srcs_version = "PY3",
)

py_test(
    name = "dp_query_test",
    srcs = ["dp_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":no_privacy_query"],
)

py_library(
    name = "discrete_gaussian_utils",
    srcs = ["discrete_gaussian_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "discrete_gaussian_utils_test",
    srcs = ["discrete_gaussian_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":discrete_gaussian_utils"],
)

py_library(
    name = "discrete_gaussian_query",
    srcs = ["discrete_gaussian_query.py"],
    srcs_version = "PY3",
    deps = [
        ":discrete_gaussian_utils",
        ":dp_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "discrete_gaussian_query_test",
    srcs = ["discrete_gaussian_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":discrete_gaussian_query",
        ":discrete_gaussian_utils",
        ":test_utils",
    ],
)

py_library(
    name = "distributed_discrete_gaussian_query",
    srcs = ["distributed_discrete_gaussian_query.py"],
    srcs_version = "PY3",
    deps = [
        ":discrete_gaussian_utils",
        ":dp_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "distributed_discrete_gaussian_query_test",
    srcs = ["distributed_discrete_gaussian_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":discrete_gaussian_utils",
        ":distributed_discrete_gaussian_query",
        ":test_utils",
    ],
)

py_library(
    name = "distributed_skellam_query",
    srcs = ["distributed_skellam_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        ":normalized_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "distributed_skellam_query_test",
    srcs = ["distributed_skellam_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distributed_skellam_query",
        ":test_utils",
    ],
)

py_library(
    name = "gaussian_query",
    srcs = ["gaussian_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "gaussian_query_test",
    size = "small",
    srcs = ["gaussian_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gaussian_query",
        ":test_utils",
    ],
)

py_library(
    name = "no_privacy_query",
    srcs = ["no_privacy_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "no_privacy_query_test",
    size = "small",
    srcs = ["no_privacy_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":no_privacy_query",
        ":test_utils",
    ],
)

py_library(
    name = "normalized_query",
    srcs = ["normalized_query.py"],
    srcs_version = "PY3",
    deps = [":dp_query"],
)

py_test(
    name = "normalized_query_test",
    size = "small",
    srcs = ["normalized_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gaussian_query",
        ":normalized_query",
        ":test_utils",
    ],
)

py_library(
    name = "nested_query",
    srcs = ["nested_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "nested_query_test",
    size = "small",
    srcs = ["nested_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        ":gaussian_query",
        ":nested_query",
        ":normalized_query",
        ":test_utils",
    ],
)

py_library(
    name = "quantile_adaptive_clip_sum_query",
    srcs = ["quantile_adaptive_clip_sum_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        ":gaussian_query",
        ":quantile_estimator_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "quantile_adaptive_clip_sum_query_test",
    srcs = ["quantile_adaptive_clip_sum_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":quantile_adaptive_clip_sum_query",
        ":test_utils",
    ],
)

py_library(
    name = "quantile_estimator_query",
    srcs = ["quantile_estimator_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        ":gaussian_query",
        ":no_privacy_query",
        ":normalized_query",
        ":tree_aggregation_query",
    ],
)

py_test(
    name = "quantile_estimator_query_test",
    srcs = ["quantile_estimator_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":quantile_estimator_query",
        ":test_utils",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tree_aggregation",
    srcs = ["tree_aggregation.py"],
    srcs_version = "PY3",
)

py_test(
    name = "tree_aggregation_test",
    srcs = ["tree_aggregation_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [":tree_aggregation"],
)

py_test(
    name = "tree_aggregation_query_test",
    srcs = ["tree_aggregation_query_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        ":tree_aggregation",
        ":tree_aggregation_query",
    ],
)

py_library(
    name = "tree_aggregation_query",
    srcs = ["tree_aggregation_query.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_query",
        ":tree_aggregation",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_library(
    name = "tree_range_query",
    srcs = ["tree_range_query.py"],
    srcs_version = "PY3",
    deps = [
        ":distributed_discrete_gaussian_query",
        ":dp_query",
        ":gaussian_query",
        "@com_google_differential_py//dp_accounting:accounting",
    ],
)

py_test(
    name = "tree_range_query_test",
    srcs = ["tree_range_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tree_range_query"],
)

py_test(
    name = "restart_query_test",
    srcs = ["restart_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":restart_query",
        ":tree_aggregation_query",
    ],
)

py_library(
    name = "restart_query",
    srcs = ["restart_query.py"],
    srcs_version = "PY3",
    deps = [":dp_query"],
)
