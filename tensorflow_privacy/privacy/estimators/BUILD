load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "estimators",
    srcs = ["__init__.py"],
)

py_library(
    name = "head_utils",
    srcs = [
        "head_utils.py",
    ],
    deps = [
        ":binary_class_head",
        ":multi_class_head",
    ],
)

py_library(
    name = "binary_class_head",
    srcs = [
        "binary_class_head.py",
    ],
)

py_library(
    name = "multi_class_head",
    srcs = [
        "multi_class_head.py",
    ],
)

py_library(
    name = "multi_label_head",
    srcs = [
        "multi_label_head.py",
    ],
)

py_library(
    name = "dnn",
    srcs = [
        "dnn.py",
    ],
    deps = [":head_utils"],
)

py_library(
    name = "test_utils",
    srcs = [
        "test_utils.py",
    ],
)

py_test(
    name = "binary_class_head_test",
    timeout = "long",
    srcs = ["binary_class_head_test.py"],
    deps = [
        ":binary_class_head",
        ":test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "multi_class_head_test",
    timeout = "long",
    srcs = ["multi_class_head_test.py"],
    deps = [
        ":multi_class_head",
        ":test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "multi_label_head_test",
    timeout = "long",
    srcs = ["multi_label_head_test.py"],
    deps = [
        ":multi_label_head",
        ":test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)

py_test(
    name = "dnn_test",
    timeout = "long",
    srcs = ["dnn_test.py"],
    deps = [
        ":dnn",
        ":test_utils",
        "//tensorflow_privacy/privacy/optimizers:dp_optimizer_keras",
    ],
)
