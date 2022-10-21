load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
)

py_library(
    name = "clip_and_aggregate_gradients",
    srcs = [
        "clip_and_aggregate_gradients.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "dp_optimizer",
    srcs = [
        "dp_optimizer.py",
    ],
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/dp_query:gaussian_query"],
)

py_library(
    name = "dp_optimizer_factory",
    srcs = [
        "dp_optimizer_keras.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/dp_query",
        "//tensorflow_privacy/privacy/dp_query:gaussian_query",
        "//tensorflow_privacy/privacy/dp_query:restart_query",
        "//tensorflow_privacy/privacy/dp_query:tree_aggregation_query",
    ],
)

py_library(
    name = "dp_optimizer_keras_sparse",
    srcs = [
        "dp_optimizer_keras_sparse.py",
    ],
    srcs_version = "PY3",
    deps = [":clip_and_aggregate_gradients"],
)

py_library(
    name = "dp_optimizer_vectorized",
    srcs = [
        "dp_optimizer_vectorized.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "dp_optimizer_keras",
    srcs = [
        "dp_optimizer_keras.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/dp_query",
        "//tensorflow_privacy/privacy/dp_query:gaussian_query",
        "//tensorflow_privacy/privacy/dp_query:restart_query",
        "//tensorflow_privacy/privacy/dp_query:tree_aggregation_query",
    ],
)

py_library(
    name = "dp_optimizer_keras_vectorized",
    srcs = [
        "dp_optimizer_keras_vectorized.py",
    ],
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/dp_query:gaussian_query"],
)

py_test(
    name = "clip_and_aggregate_gradients_test",
    srcs = ["clip_and_aggregate_gradients_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":clip_and_aggregate_gradients"],
)

py_test(
    name = "dp_optimizer_test",
    timeout = "long",
    srcs = ["dp_optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dp_optimizer",
        "//tensorflow_privacy/privacy/dp_query:gaussian_query",
    ],
)

py_test(
    name = "dp_optimizer_keras_sparse_test",
    timeout = "long",
    srcs = ["dp_optimizer_keras_sparse_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":dp_optimizer_keras_sparse"],
)

py_test(
    name = "dp_optimizer_vectorized_test",
    timeout = "long",
    srcs = ["dp_optimizer_vectorized_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":dp_optimizer_vectorized"],
)

py_test(
    name = "dp_optimizer_eager_test",
    timeout = "long",
    srcs = ["dp_optimizer_eager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dp_optimizer",
        "//tensorflow_privacy/privacy/dp_query:gaussian_query",
    ],
)

py_test(
    name = "dp_optimizer_keras_test",
    timeout = "long",
    srcs = ["dp_optimizer_keras_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dp_optimizer_keras",
        ":dp_optimizer_keras_vectorized",
    ],
)
