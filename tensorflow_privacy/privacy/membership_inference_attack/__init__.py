# Copyright 2020, The TensorFlow Authors.
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
"""The old location of Membership Inference Attack sources."""

import warnings

warnings.warn(
    "\nMembership inference attack sources were moved. Please replace"
    "\nimport tensorflow_privacy.privacy.membership_inference_attack\n"
    "\nwith"
    "\nimport tensorflow_privacy.privacy.privacy_tests.membership_inference_attack"
)
