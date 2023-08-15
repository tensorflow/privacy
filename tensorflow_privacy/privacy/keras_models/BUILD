package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "keras_models",
    srcs = ["__init__.py"],
)

py_library(
    name = "dp_keras_model",
    srcs = [
        "dp_keras_model.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:clip_grads",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:common_manip_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:gradient_clipping_utils",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)

py_test(
    name = "dp_keras_model_test",
    srcs = ["dp_keras_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
        "//tensorflow_privacy/privacy/keras_models:dp_keras_model",
    ],
)

py_test(
    name = "dp_keras_model_distributed_test",
    srcs = ["dp_keras_model_distributed_test.py"],
    deps = [
        ":dp_keras_model",
        "//tensorflow_privacy/privacy/fast_gradient_clipping:layer_registry",
    ],
)
