package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "sparse_noise_utils",
    srcs = ["sparse_noise_utils.py"],
    deps = [
        ":type_aliases",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:gradient_clipping_utils",
    ],
)

py_test(
    name = "sparse_noise_utils_test",
    srcs = ["sparse_noise_utils_test.py"],
    deps = [
        ":sparse_noise_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:gradient_clipping_utils",
    ],
)

py_library(
    name = "type_aliases",
    srcs = ["type_aliases.py"],
)

py_library(
    name = "layer_registry",
    srcs = ["layer_registry.py"],
    deps = [
        ":type_aliases",
        "//tensorflow_privacy/privacy/sparsity_preserving_noise/registry_functions:embedding",
    ],
)
