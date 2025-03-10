package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "__init__",  # There is already a dp_query.
    srcs = ["__init__.py"],
)

py_library(
    name = "dp_query",
    srcs = ["dp_query.py"],
)

py_test(
    name = "dp_query_test",
    srcs = ["dp_query_test.py"],
    deps = [":no_privacy_query"],
)

py_library(
    name = "discrete_gaussian_utils",
    srcs = ["discrete_gaussian_utils.py"],
)

py_test(
    name = "discrete_gaussian_utils_test",
    srcs = ["discrete_gaussian_utils_test.py"],
    deps = [":discrete_gaussian_utils"],
)

py_library(
    name = "discrete_gaussian_query",
    srcs = ["discrete_gaussian_query.py"],
    deps = [
        ":discrete_gaussian_utils",
        ":dp_query",
    ],
)

py_test(
    name = "discrete_gaussian_query_test",
    srcs = ["discrete_gaussian_query_test.py"],
    deps = [
        ":discrete_gaussian_query",
        ":discrete_gaussian_utils",
        ":test_utils",
    ],
)

py_library(
    name = "distributed_discrete_gaussian_query",
    srcs = ["distributed_discrete_gaussian_query.py"],
    deps = [
        ":discrete_gaussian_utils",
        ":dp_query",
    ],
)

py_test(
    name = "distributed_discrete_gaussian_query_test",
    srcs = ["distributed_discrete_gaussian_query_test.py"],
    deps = [
        ":discrete_gaussian_utils",
        ":distributed_discrete_gaussian_query",
        ":test_utils",
    ],
)

py_library(
    name = "distributed_skellam_query",
    srcs = ["distributed_skellam_query.py"],
    deps = [
        ":dp_query",
        ":normalized_query",
    ],
)

py_test(
    name = "distributed_skellam_query_test",
    srcs = ["distributed_skellam_query_test.py"],
    deps = [
        ":distributed_skellam_query",
        ":test_utils",
    ],
)

py_library(
    name = "gaussian_query",
    srcs = ["gaussian_query.py"],
    deps = [":dp_query"],
)

py_test(
    name = "gaussian_query_test",
    size = "small",
    srcs = ["gaussian_query_test.py"],
    deps = [
        ":gaussian_query",
        ":test_utils",
    ],
)

py_library(
    name = "no_privacy_query",
    srcs = ["no_privacy_query.py"],
    deps = [":dp_query"],
)

py_test(
    name = "no_privacy_query_test",
    size = "small",
    srcs = ["no_privacy_query_test.py"],
    deps = [
        ":no_privacy_query",
        ":test_utils",
    ],
)

py_library(
    name = "normalized_query",
    srcs = ["normalized_query.py"],
    deps = [":dp_query"],
)

py_test(
    name = "normalized_query_test",
    size = "small",
    srcs = ["normalized_query_test.py"],
    deps = [
        ":gaussian_query",
        ":normalized_query",
        ":test_utils",
    ],
)

py_library(
    name = "nested_query",
    srcs = ["nested_query.py"],
    deps = [":dp_query"],
)

py_test(
    name = "nested_query_test",
    size = "small",
    srcs = ["nested_query_test.py"],
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
    deps = [
        ":dp_query",
        ":gaussian_query",
        ":quantile_estimator_query",
    ],
)

py_test(
    name = "quantile_adaptive_clip_sum_query_test",
    srcs = ["quantile_adaptive_clip_sum_query_test.py"],
    deps = [
        ":quantile_adaptive_clip_sum_query",
        ":test_utils",
    ],
)

py_library(
    name = "quantile_estimator_query",
    srcs = ["quantile_estimator_query.py"],
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
    deps = [
        ":quantile_estimator_query",
        ":test_utils",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

py_library(
    name = "tree_aggregation",
    srcs = ["tree_aggregation.py"],
)

py_test(
    name = "tree_aggregation_test",
    srcs = ["tree_aggregation_test.py"],
    shard_count = 10,
    deps = [":tree_aggregation"],
)

py_test(
    name = "tree_aggregation_query_test",
    srcs = ["tree_aggregation_query_test.py"],
    shard_count = 5,
    deps = [
        ":test_utils",
        ":tree_aggregation",
        ":tree_aggregation_query",
    ],
)

py_library(
    name = "tree_aggregation_query",
    srcs = ["tree_aggregation_query.py"],
    deps = [
        ":dp_query",
        ":tree_aggregation",
    ],
)

py_library(
    name = "tree_range_query",
    srcs = ["tree_range_query.py"],
    deps = [
        ":distributed_discrete_gaussian_query",
        ":dp_query",
        ":gaussian_query",
    ],
)

py_test(
    name = "tree_range_query_test",
    srcs = ["tree_range_query_test.py"],
    deps = [":tree_range_query"],
)

py_test(
    name = "restart_query_test",
    srcs = ["restart_query_test.py"],
    deps = [
        ":restart_query",
        ":tree_aggregation_query",
    ],
)

py_library(
    name = "restart_query",
    srcs = ["restart_query.py"],
    deps = [":dp_query"],
)

py_library(
    name = "quantile_adaptive_clip_tree_query",
    srcs = ["quantile_adaptive_clip_tree_query.py"],
    deps = [
        ":dp_query",
        ":quantile_estimator_query",
        ":tree_aggregation_query",
    ],
)

py_test(
    name = "quantile_adaptive_clip_tree_query_test",
    timeout = "long",
    srcs = ["quantile_adaptive_clip_tree_query_test.py"],
    shard_count = 5,
    deps = [
        ":quantile_adaptive_clip_tree_query",
        ":test_utils",
    ],
)
