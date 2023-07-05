load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "dense",
    srcs = ["dense.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "dense_test",
    size = "large",
    srcs = ["dense_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "embedding",
    srcs = ["embedding.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases"],
)

py_test(
    name = "embedding_test",
    size = "large",
    srcs = ["embedding_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)
