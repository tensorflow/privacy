package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "einsum_utils",
    srcs = ["einsum_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "einsum_utils_test",
    srcs = ["einsum_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":einsum_utils"],
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
    shard_count = 12,
    srcs_version = "PY3",
    deps = [
        ":dense",
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
    srcs = ["embedding_test.py"],
    python_version = "PY3",
    shard_count = 12,
    srcs_version = "PY3",
    deps = [
        ":dense",
        ":embedding",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "layer_normalization",
    srcs = ["layer_normalization.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases"],
)

py_test(
    name = "layer_normalization_test",
    srcs = ["layer_normalization_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        ":dense",
        ":layer_normalization",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)
