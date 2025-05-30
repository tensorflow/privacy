package(default_visibility = ["//visibility:public"])

py_library(
    name = "type_aliases",
    srcs = ["type_aliases.py"],
)

py_library(
    name = "bert_encoder_utils",
    srcs = ["bert_encoder_utils.py"],
    deps = [":gradient_clipping_utils"],
)

py_test(
    name = "bert_encoder_utils_test",
    srcs = ["bert_encoder_utils_test.py"],
    deps = [":bert_encoder_utils"],
)

py_library(
    name = "common_manip_utils",
    srcs = ["common_manip_utils.py"],
    deps = [":type_aliases"],
)

py_library(
    name = "common_test_utils",
    srcs = ["common_test_utils.py"],
    deps = [
        ":clip_grads",
        ":layer_registry",
        ":type_aliases",
    ],
)

py_library(
    name = "gradient_clipping_utils",
    srcs = ["gradient_clipping_utils.py"],
    deps = [
        ":common_manip_utils",
        ":layer_registry",
        ":type_aliases",
        "//tensorflow_privacy/privacy/sparsity_preserving_noise:layer_registry",
        "//tensorflow_privacy/privacy/sparsity_preserving_noise:type_aliases",
    ],
)

py_test(
    name = "gradient_clipping_utils_test",
    srcs = ["gradient_clipping_utils_test.py"],
    shard_count = 8,
    deps = [
        ":gradient_clipping_utils",
        ":layer_registry",
        "//tensorflow_privacy/privacy/sparsity_preserving_noise:layer_registry",
    ],
)

py_library(
    name = "layer_registry",
    srcs = ["layer_registry.py"],
    deps = [
        ":type_aliases",
        "//tensorflow_privacy/privacy/fast_gradient_clipping/registry_functions:dense",
        "//tensorflow_privacy/privacy/fast_gradient_clipping/registry_functions:embedding",
    ],
)

py_library(
    name = "clip_grads",
    srcs = ["clip_grads.py"],
    deps = [
        ":common_manip_utils",
        ":gradient_clipping_utils",
        ":layer_registry",
        ":type_aliases",
    ],
)

py_library(
    name = "noise_utils",
    srcs = ["noise_utils.py"],
    deps = ["//tensorflow_privacy/privacy/sparsity_preserving_noise:sparse_noise_utils"],
)

py_test(
    name = "clip_grads_test",
    srcs = ["clip_grads_test.py"],
    shard_count = 8,
    deps = [
        ":clip_grads",
        ":common_test_utils",
        ":gradient_clipping_utils",
        ":layer_registry",
        ":type_aliases",
    ],
)

py_test(
    name = "noise_utils_test",
    srcs = ["noise_utils_test.py"],
    deps = [":noise_utils"],
)
