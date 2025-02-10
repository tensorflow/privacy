package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "registry_function_utils",
    srcs = ["registry_function_utils.py"],
    deps = ["//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases"],
)

py_library(
    name = "einsum_utils",
    srcs = ["einsum_utils.py"],
    deps = ["//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils"],
)

py_test(
    name = "einsum_utils_test",
    srcs = ["einsum_utils_test.py"],
    shard_count = 4,
    deps = [":einsum_utils"],
)

py_library(
    name = "einsum_dense",
    srcs = ["einsum_dense.py"],
    deps = [
        ":einsum_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "einsum_dense_test",
    size = "large",
    srcs = ["einsum_dense_test.py"],
    shard_count = 12,
    deps = [
        ":dense",
        ":einsum_dense",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "dense",
    srcs = ["dense.py"],
    deps = [
        ":einsum_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "dense_test",
    size = "large",
    srcs = ["dense_test.py"],
    shard_count = 12,
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
    deps = [
        ":registry_function_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    shard_count = 12,
    deps = [
        ":dense",
        ":embedding",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "nlp_on_device_embedding",
    srcs = ["nlp_on_device_embedding.py"],
    deps = [
        ":registry_function_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "nlp_on_device_embedding_test",
    srcs = ["nlp_on_device_embedding_test.py"],
    shard_count = 6,
    deps = [
        ":dense",
        ":nlp_on_device_embedding",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "nlp_position_embedding",
    srcs = ["nlp_position_embedding.py"],
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "nlp_position_embedding_test",
    srcs = ["nlp_position_embedding_test.py"],
    shard_count = 6,
    deps = [
        ":dense",
        ":nlp_position_embedding",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "layer_normalization",
    srcs = ["layer_normalization.py"],
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "layer_normalization_test",
    srcs = ["layer_normalization_test.py"],
    shard_count = 8,
    deps = [
        ":dense",
        ":layer_normalization",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_library(
    name = "multi_head_attention",
    srcs = ["multi_head_attention.py"],
    deps = [
        ":einsum_dense",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:type_aliases",
    ],
)

py_test(
    name = "multi_head_attention_test",
    srcs = ["multi_head_attention_test.py"],
    shard_count = 8,
    deps = [
        ":dense",
        ":multi_head_attention",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_test_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)
