# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey, canonicalize_scene_url, require_scene_id


@pytest.mark.parametrize(
    ("first_query", "second_query"),
    [
        (
            "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260722T000000Z&X-Amz-Signature=one",
            "X-Amz-Date=20260723T000000Z&X-Amz-Signature=two&X-Amz-Expires=3600",
        ),
        (
            "X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Date=20260722T000000Z&X-Goog-Signature=one",
            "X-Goog-Date=20260723T000000Z&X-Goog-Signature=two&X-Goog-Expires=3600",
        ),
        (
            "temp_url_sig=one&temp_url_expires=1784678400",
            "temp_url_expires=1784764800&temp_url_sig=two",
        ),
    ],
)
def test_rotating_presigned_credentials_do_not_change_scene_cache_key(first_query: str, second_query: str):
    first = f"https://EXAMPLE.com/scenes/a.usdz?{first_query}&versionId=7"
    second = f"https://example.com/scenes/a.usdz?versionId=7&{second_query}"

    first_key = SceneCacheKey.from_url("scene", first)
    second_key = SceneCacheKey.from_url("scene", second)

    assert first_key.canonical_url == "https://example.com/scenes/a.usdz?versionId=7"
    assert first_key == second_key
    assert first_key.key_hash == second_key.key_hash


def test_object_version_and_path_remain_part_of_scene_cache_key():
    base = SceneCacheKey.from_url("scene", "https://example.com/a.usdz?versionId=1")
    new_version = SceneCacheKey.from_url("scene", "https://example.com/a.usdz?versionId=2")
    new_path = SceneCacheKey.from_url("scene", "https://example.com/b.usdz?versionId=1")

    assert len({base.key_hash, new_version.key_hash, new_path.key_hash}) == 3


def test_scene_id_remains_part_of_scene_cache_key():
    url = "https://example.com/a.usdz"
    assert SceneCacheKey.from_url("scene-a", url).key_hash != SceneCacheKey.from_url("scene-b", url).key_hash


def test_url_and_url_less_keys_differ():
    url_key = SceneCacheKey.from_url("scene", "https://example.com/a.usdz")
    url_less_key = SceneCacheKey(scene_id="scene")

    assert url_key.key_hash.startswith("scene-")
    assert url_less_key.key_hash.startswith("scene-")
    assert url_less_key.key_hash != url_key.key_hash


@pytest.mark.parametrize("scene_url", ["", "local", "relative/path.usdz", "s3://example.com/a.usdz"])
def test_empty_relative_and_unsupported_scene_urls_are_rejected(scene_url: str):
    with pytest.raises(ValueError, match="Scene URL|scene URL"):
        SceneCacheKey.from_url("scene", scene_url)


def test_content_sha_changes_backend_cache_key_but_not_scene_cache_key():
    scene_key = SceneCacheKey.from_url("scene", "https://example.com/a.usdz")
    first = BackendCacheKey(scene_key, "a" * 64)
    second = BackendCacheKey(scene_key, "b" * 64)

    assert first.key_hash != second.key_hash
    assert first.scene_key == second.scene_key
    assert first == BackendCacheKey(scene_key, "A" * 64)


def test_local_path_and_percent_encoded_file_uri_have_same_canonical_url(tmp_path: Path):
    scene = tmp_path / "scene with spaces.usdz"
    assert canonicalize_scene_url(str(scene)) == canonicalize_scene_url(scene.as_uri())


def test_missing_scene_id_is_rejected():
    with pytest.raises(ValueError, match="x-nre-scene-id"):
        require_scene_id(None)
