workspace(name = "org_tensorflow_privacy")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.0.3",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.5.0",
)

git_repository(
  name = "com_google_differential_py",
  remote = "https://github.com/google/differential-privacy.git",
  commit = "8536a3af6b147b1cce6f884826bfd5f2009ae50f",
)

# This is a workaround until the @com_google_differential_py WORKSPACE declares
# the nested differential-privacy/python/WORKSPACE as a local_repository, or
# https://github.com/bazelbuild/bazel/issues/1943 is fixed to support recursive
# WORKSPACE loading via git_repository.
load("@com_google_differential_py//python:accounting_py_deps.bzl", "accounting_py_deps")
accounting_py_deps()
# We can't directly use accounting_py_deps_init.bzl because it hard-codes a path
# to the requirements file that is relative to the workspace root.
load("@rules_python//python:pip.bzl", "pip_install")
pip_install(
  name = "accounting_py_pip_deps",
  requirements = "@com_google_differential_py//python:requirements.txt",
)
