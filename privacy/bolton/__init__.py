import sys
from distutils.version import LooseVersion
import tensorflow as tf

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  raise ImportError("Please upgrade your version of tensorflow from: {0} "
                    "to at least 2.0.0 to use privacy/bolton".format(
    LooseVersion(tf.__version__)))
if hasattr(sys, 'skip_tf_privacy_import'):  # Useful for standalone scripts.
  pass
else:
  from privacy.bolton.models import BoltonModel
  from privacy.bolton.optimizers import Bolton
  from privacy.bolton.losses import StrongConvexHuber
  from privacy.bolton.losses import StrongConvexBinaryCrossentropy