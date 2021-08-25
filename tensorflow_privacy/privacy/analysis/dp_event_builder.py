# Copyright 2021, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Builder class for ComposedDpEvent."""

import collections

from tensorflow_privacy.privacy.analysis import dp_event


class DpEventBuilder(object):
  """Constructs a `DpEvent` representing the composition of a series of events.

  Two common use cases of the `DpEventBuilder` are 1) for producing and tracking
  a ledger of `DpEvent`s during sequential accounting using a
  `PrivacyAccountant`, and 2) for building up a description of a composite
  mechanism for subsequent batch accounting.
  """

  def __init__(self):
    self._events = collections.OrderedDict()
    self._composed_event = None

  def compose(self, event: dp_event.DpEvent, count: int = 1):
    """Composes new event into event represented by builder.

    Args:
      event: The new event to compose.
      count: The number of times to compose the event.
    """
    if not isinstance(event, dp_event.DpEvent):
      raise TypeError('`event` must be a subclass of `DpEvent`. '
                      f'Found {type(event)}.')
    if not isinstance(count, int):
      raise TypeError(f'`count` must be an integer. Found {type(count)}.')
    if count < 1:
      raise ValueError(f'`count` must be positive. Found {count}.')

    if isinstance(event, dp_event.ComposedDpEvent):
      for composed_event in event.events:
        self.compose(composed_event, count)
    elif isinstance(event, dp_event.SelfComposedDpEvent):
      self.compose(event.event, count * event.count)
    elif isinstance(event, dp_event.NoOpDpEvent):
      return
    else:
      current_count = self._events.get(event, 0)
      self._events[event] = current_count + count
      self._composed_event = None

  def build(self) -> dp_event.DpEvent:
    """Builds and returns the composed DpEvent represented by the builder."""
    if not self._composed_event:
      self_composed_events = []
      for event, count in self._events.items():
        if count == 1:
          self_composed_events.append(event)
        else:
          self_composed_events.append(
              dp_event.SelfComposedDpEvent(event, count))
      if not self_composed_events:
        self._composed_event = dp_event.NoOpDpEvent()
      elif len(self_composed_events) == 1:
        self._composed_event = self_composed_events[0]
      else:
        self._composed_event = dp_event.ComposedDpEvent(self_composed_events)

    return self._composed_event
