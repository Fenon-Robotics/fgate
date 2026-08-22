from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from qc_pipeline.storage import ObjectNotFound, R2Store, StorageError


class HeadClient:
    def __init__(self, status: int):
        self.status = status

    def head_object(self, **kwargs):
        code = "404" if self.status == 404 else "AccessDenied"
        raise ClientError(
            {
                "Error": {"Code": code, "Message": "failure"},
                "ResponseMetadata": {"HTTPStatusCode": self.status},
            },
            "HeadObject",
        )


def store_with_client(client) -> R2Store:
    store = object.__new__(R2Store)
    store.client = client
    return store


def test_head_distinguishes_missing_from_permission_failure() -> None:
    with pytest.raises(ObjectNotFound):
        store_with_client(HeadClient(404)).head("bucket", "missing")
    with pytest.raises(StorageError, match="HEAD failed"):
        store_with_client(HeadClient(403)).head("bucket", "denied")


def test_upload_does_not_treat_head_failure_as_missing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "report.json"
    source.write_text("{}")
    store = store_with_client(object())

    def fail_head(bucket: str, key: str):
        raise StorageError("network unavailable")

    monkeypatch.setattr(store, "head", fail_head)
    with pytest.raises(StorageError, match="network unavailable"):
        store.upload_create_only(
            "bucket",
            "report.json",
            source,
            content_type="application/json",
            sha256="digest",
        )
