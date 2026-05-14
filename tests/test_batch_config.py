import unittest
from contextlib import redirect_stdout
from io import StringIO

from soma_retargeter.pipelines.batch_config import (
    CONTACT_AWARE_BATCH_SIZE_LOG,
    resolve_retarget_batch_size,
)


class TestBatchConfig(unittest.TestCase):
    def test_contact_aware_ik_forces_batch_size_one(self):
        config = {"contact_aware_foot_ik": {"enabled": True}}

        out = StringIO()
        with redirect_stdout(out):
            batch_size = resolve_retarget_batch_size(100, config)

        self.assertEqual(batch_size, 1)
        self.assertIn(CONTACT_AWARE_BATCH_SIZE_LOG, out.getvalue())

    def test_contact_source_none_keeps_configured_batch_size(self):
        config = {"contact_aware_foot_ik": {"enabled": True, "contact_source": "none"}}

        batch_size = resolve_retarget_batch_size(100, config, log=False)

        self.assertEqual(batch_size, 100)

    def test_disabled_contact_aware_ik_keeps_configured_batch_size(self):
        config = {"contact_aware_foot_ik": {"enabled": False}}

        batch_size = resolve_retarget_batch_size(100, config, log=False)

        self.assertEqual(batch_size, 100)


if __name__ == "__main__":
    unittest.main()
