load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "__init__",  # There is already a membership_inference_attack.
    srcs = ["__init__.py"],
)

py_library(
    name = "plotting",
    srcs = ["plotting.py"],
    srcs_version = "PY3",
)

py_test(
    name = "membership_inference_attack_test",
    timeout = "long",
    srcs = ["membership_inference_attack_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":membership_inference_attack"],
)

py_test(
    name = "data_structures_test",
    timeout = "long",
    srcs = ["data_structures_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":membership_inference_attack",
        "//tensorflow_privacy/privacy/privacy_tests:utils",
    ],
)

py_test(
    name = "dataset_slicing_test",
    srcs = ["dataset_slicing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":membership_inference_attack"],
)

py_test(
    name = "models_test",
    timeout = "long",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":membership_inference_attack"],
)

py_test(
    name = "privacy_report_test",
    timeout = "long",
    srcs = ["privacy_report_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":membership_inference_attack",
        ":privacy_report",
    ],
)

py_library(
    name = "privacy_report",
    srcs = ["privacy_report.py"],
    srcs_version = "PY3",
    deps = [":membership_inference_attack"],
)

py_library(
    name = "membership_inference_attack",
    srcs = [
        "data_structures.py",
        "dataset_slicing.py",
        "membership_inference_attack.py",
        "models.py",
        "plotting.py",
        "seq2seq_mia.py",
    ],
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/privacy_tests:utils"],
)

py_library(
    name = "utils_tensorboard",
    srcs = ["utils_tensorboard.py"],
    srcs_version = "PY3",
    deps = [":membership_inference_attack"],
)

py_test(
    name = "tf_estimator_evaluation_test",
    srcs = ["tf_estimator_evaluation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":membership_inference_attack",
        ":tf_estimator_evaluation",
    ],
)

py_library(
    name = "tf_estimator_evaluation",
    srcs = ["tf_estimator_evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":membership_inference_attack",
        ":utils_tensorboard",
        "//tensorflow_privacy/privacy/privacy_tests:utils",
    ],
)

py_binary(
    name = "tf_estimator_evaluation_example",
    srcs = ["tf_estimator_evaluation_example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":membership_inference_attack",
        ":tf_estimator_evaluation",
    ],
)

py_library(
    name = "keras_evaluation",
    srcs = ["keras_evaluation.py"],
    srcs_version = "PY3",
    deps = [
        ":membership_inference_attack",
        ":utils_tensorboard",
        "//tensorflow_privacy/privacy/privacy_tests:utils",
    ],
)

py_test(
    name = "seq2seq_mia_test",
    srcs = ["seq2seq_mia_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":membership_inference_attack"],
)

py_binary(
    name = "keras_evaluation_example",
    srcs = ["keras_evaluation_example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_evaluation",
        ":membership_inference_attack",
    ],
)

py_test(
    name = "keras_evaluation_test",
    srcs = ["keras_evaluation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_evaluation",
        ":membership_inference_attack",
    ],
)

py_library(
    name = "advanced_mia",
    srcs = [
        "advanced_mia.py",
    ],
    srcs_version = "PY3",
    deps = ["//tensorflow_privacy/privacy/privacy_tests:utils"],
)

py_test(
    name = "advanced_mia_test",
    timeout = "long",
    srcs = ["advanced_mia_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":advanced_mia"],
)

py_binary(
    name = "advanced_mia_example",
    srcs = ["advanced_mia_example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":advanced_mia",
        ":membership_inference_attack",
        "//tensorflow_privacy/privacy/privacy_tests:utils",
    ],
)
