load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "__init__",  # There is already a membership_inference_attack.
    srcs = ["__init__.py"],
)

py_library(
    name = "keras_evaluation",
    srcs = ["keras_evaluation.py"],
    deps = ["//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack:keras_evaluation"],
)

py_library(
    name = "plotting",
    srcs = ["plotting.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack:plotting",
    ],
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
    deps = [
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack",
    ],
)

py_library(
    name = "privacy_report",
    srcs = ["privacy_report.py"],
    deps = ["//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack"],
)

py_library(
    name = "tf_estimator_evaluation",
    srcs = ["tf_estimator_evaluation.py"],
    deps = ["//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack:tf_estimator_evaluation"],
)
