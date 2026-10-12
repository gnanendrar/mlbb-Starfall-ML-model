import json
import os
import pytest
from starfall.control_lease import ControlLease


def test_second_controller_is_rejected_and_clean_exit_releases(tmp_path):
    path = tmp_path / "control.lock"
    lease = ControlLease(path).acquire()
    assert json.loads(path.read_text())["pid"] == os.getpid()
    with pytest.raises(RuntimeError, match="Another phone controller"):
        ControlLease(path).acquire()
    lease.close()
    assert not path.exists()
