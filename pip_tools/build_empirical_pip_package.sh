#!/usr/bin/env bash
# Copyright 2020, The TensorFlow Privacy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tool to build the TensorFlow Privacy/Privacy Tests pip package.
set -e

main() {
  # Create a working directory.
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT

  # Create a virtual environment
  python3.11 -m venv "${temp_dir}/venv"
  source "${temp_dir}/venv/bin/activate"
  python --version
  pip install --upgrade pip
  pip --version

  # Build the pip package
  pip install --upgrade setuptools wheel
  python "setup_empirical.py" sdist bdist_wheel

  # Cleanup.
  deactivate
}

main "$@"
