from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class StorageError(RuntimeError):
    pass


class SourceConflict(StorageError):
    pass


class DestinationConflict(StorageError):
    pass


class ObjectNotFound(StorageError):
    pass


@dataclass(frozen=True)
class ObjectIdentity:
    bucket: str
    key: str
    size_bytes: int
    etag: str
    metadata: dict[str, str]


class ObjectStore(Protocol):
    def head(self, bucket: str, key: str) -> ObjectIdentity: ...

    def download(
        self, bucket: str, key: str, destination: Path, *, expected_size: int, expected_etag: str
    ) -> ObjectIdentity: ...

    def upload_create_only(
        self, bucket: str, key: str, source: Path, *, content_type: str, sha256: str
    ) -> ObjectIdentity: ...

    def get_json(self, bucket: str, key: str) -> dict[str, Any]: ...

    def presign_get(self, bucket: str, key: str, *, expires_seconds: int = 3600) -> str: ...


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise StorageError(f"environment file not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name.strip()] = value
    required = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL")
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise StorageError(f"missing R2 settings: {', '.join(missing)}")
    return values


def normalize_etag(value: str | None) -> str:
    return (value or "").strip().strip('"')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class R2Store:
    def __init__(self, env_file: Path):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise StorageError("boto3 is required for R2") from error
        values = read_env(env_file)
        self.client = boto3.client(
            "s3",
            endpoint_url=values["R2_ENDPOINT_URL"],
            aws_access_key_id=values["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                max_pool_connections=64,
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 8, "mode": "standard"},
            ),
        )

    @staticmethod
    def _identity(bucket: str, key: str, response: dict[str, Any]) -> ObjectIdentity:
        return ObjectIdentity(
            bucket=bucket,
            key=key,
            size_bytes=int(response.get("ContentLength", 0)),
            etag=normalize_etag(response.get("ETag")),
            metadata={str(k): str(v) for k, v in (response.get("Metadata") or {}).items()},
        )

    def head(self, bucket: str, key: str) -> ObjectIdentity:
        from botocore.exceptions import ClientError

        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"NoSuchKey", "NotFound", "404"} or status == 404:
                raise ObjectNotFound(f"R2 object not found: {bucket}/{key}") from error
            raise StorageError(f"R2 HEAD failed for {bucket}/{key}: {error}") from error
        except Exception as error:
            raise StorageError(f"R2 HEAD failed for {bucket}/{key}: {error}") from error
        return self._identity(bucket, key, response)

    def presign_get(self, bucket: str, key: str, *, expires_seconds: int = 3600) -> str:
        try:
            value = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        except Exception as error:
            raise StorageError(f"R2 presign failed for {bucket}/{key}: {error}") from error
        return str(value)

    def download(
        self,
        bucket: str,
        key: str,
        destination: Path,
        *,
        expected_size: int,
        expected_etag: str,
    ) -> ObjectIdentity:
        identity = self.head(bucket, key)
        if identity.size_bytes != expected_size or identity.etag != normalize_etag(expected_etag):
            raise SourceConflict(
                f"source identity drift for {bucket}/{key}: "
                f"expected size={expected_size} etag={normalize_etag(expected_etag)}, "
                f"observed size={identity.size_bytes} etag={identity.etag}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        try:
            response = self.client.get_object(
                Bucket=bucket,
                Key=key,
                IfMatch=f'"{identity.etag}"',
            )
            body = response["Body"]
            with temporary.open("wb") as stream:
                while block := body.read(8 * 1024 * 1024):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size != expected_size:
                raise StorageError(
                    f"download size mismatch for {bucket}/{key}: {temporary.stat().st_size}"
                )
            temporary.replace(destination)
        except SourceConflict:
            raise
        except Exception as error:
            raise StorageError(f"R2 download failed for {bucket}/{key}: {error}") from error
        finally:
            if temporary.exists():
                temporary.unlink()
        return identity

    def upload_create_only(
        self,
        bucket: str,
        key: str,
        source: Path,
        *,
        content_type: str,
        sha256: str,
    ) -> ObjectIdentity:
        from botocore.exceptions import ClientError

        size = source.stat().st_size
        try:
            existing = self.head(bucket, key)
        except ObjectNotFound:
            existing = None
        if existing is not None:
            if existing.size_bytes == size and existing.metadata.get("sha256") == sha256:
                return existing
            raise DestinationConflict(
                f"destination already exists with different identity: {bucket}/{key}"
            )
        try:
            with source.open("rb") as stream:
                self.client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=stream,
                    ContentLength=size,
                    ContentType=content_type,
                    Metadata={"sha256": sha256},
                    IfNoneMatch="*",
                )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"PreconditionFailed", "412"} or status == 412:
                raise DestinationConflict(
                    f"destination creation raced for {bucket}/{key}"
                ) from error
            raise StorageError(f"R2 upload failed for {bucket}/{key}: {error}") from error
        except Exception as error:
            raise StorageError(f"R2 upload failed for {bucket}/{key}: {error}") from error
        verified = self.head(bucket, key)
        if verified.size_bytes != size or verified.metadata.get("sha256") != sha256:
            raise StorageError(f"uploaded object verification failed for {bucket}/{key}")
        return verified

    def get_json(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            return json.loads(response["Body"].read())
        except Exception as error:
            raise StorageError(f"R2 JSON read failed for {bucket}/{key}: {error}") from error
