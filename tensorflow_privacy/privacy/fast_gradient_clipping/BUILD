load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "gradient_clipping_utils",
    srcs = ["gradient_clipping_utils.py"],
    srcs_version = "PY3",
    deps = [":layer_registry"],
)

py_library(
    name = "layer_registry",
    srcs = ["layer_registry.py"],
    srcs_version = "PY3",
)

py_library(
    name = "clip_grads",
    srcs = ["clip_grads.py"],
    srcs_version = "PY3",
    deps = [
        ":gradient_clipping_utils",
        ":layer_registry",
    ],
)

py_test(
    name = "clip_grads_test",
    srcs = ["clip_grads_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clip_grads",
        ":layer_registry",
    ],
)
