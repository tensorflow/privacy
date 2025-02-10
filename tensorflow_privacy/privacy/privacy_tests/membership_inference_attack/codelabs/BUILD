package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "example",
    srcs = ["example.py"],
    deps = [
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack",
        "//tensorflow_privacy/privacy/privacy_tests/membership_inference_attack:privacy_report",
    ],
)
