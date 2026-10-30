# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the model registry module."""

import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import requests

from nurec.utils.model_registry import (
    GitLabModelRegistry,
    HuggingFaceModelRegistry,
    ModelRegistry,
    ModelRegistryError,
    NgcModelRegistry,
    create_model_registry,
    fetch_bundle,
    register_gitlab_domain,
)


_GITLAB_TEST_DOMAIN = "gitlab.example.com"
_GITLAB_MODEL_URL = f"https://{_GITLAB_TEST_DOMAIN}/projects/models/test/model.tar.gz"
_GITLAB_MODEL_URL_HTTP = f"http://{_GITLAB_TEST_DOMAIN}/projects/models/test/model.tar.gz"


class TestModelRegistry(unittest.TestCase):
    """Test cases for the base ModelRegistry class."""

    class MockModelRegistry(ModelRegistry):
        """Mock implementation for unit testing base class."""

        @property
        def domain(self) -> str:
            return "mock.example.com"

        @staticmethod
        def api_key_hint() -> str:
            return "Mock API Key (starting with 'mock-')"

        def _validate_url(self, url: str) -> bool:
            # Only accept URLs starting with 'mock://'
            return url.startswith("mock://")

        def _download_to_file(self, filename: str) -> str:
            """Mock implementation that just returns a predefined path"""
            return "mock_download_path"

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_init_valid_url(self):
        """Test initialization with a valid URL."""
        registry = self.MockModelRegistry(
            model_url="mock://valid/url", model_cache_dir=self.test_dir, api_key="test_key"
        )
        self.assertEqual(registry.model_url, "mock://valid/url")
        self.assertEqual(registry.model_cache_dir, self.test_dir)

    def test_init_invalid_url(self):
        """Test initialization with an invalid URL."""
        with self.assertRaises(ModelRegistryError) as cm:
            self.MockModelRegistry(model_url="invalid://url", model_cache_dir=self.test_dir, api_key="test_key")
        self.assertIn("Invalid model URL", str(cm.exception))

    def test_verify_cached_file(self):
        """Test the _verify_cached_file method."""
        # Test with valid file
        valid_file = self.test_dir / "valid_file"
        valid_file.write_text("x" * 2048)  # 2KB file, above MIN_FILE_SIZE

        # Test with invalid file
        invalid_file = self.test_dir / "invalid_file"
        invalid_file.write_text("x" * 512)  # 512B file, below MIN_FILE_SIZE

        registry = self.MockModelRegistry(
            model_url="mock://valid/url", model_cache_dir=self.test_dir, api_key="test_key"
        )

        self.assertTrue(registry._verify_cached_file(valid_file))
        self.assertFalse(registry._verify_cached_file(invalid_file))
        self.assertFalse(registry._verify_cached_file(self.test_dir / "nonexistent"))

    def test_verify_downloaded_file(self):
        """Test the _verify_downloaded_file method."""
        # Create a test file
        test_file = self.test_dir / "test_file"
        content = "x" * 2048  # 2KB file
        test_file.write_text(content)

        registry = self.MockModelRegistry(
            model_url="mock://valid/url", model_cache_dir=self.test_dir, api_key="test_key"
        )

        # Test with correct size
        self.assertTrue(registry._verify_downloaded_file(test_file, len(content)))

        # Test with incorrect size
        self.assertFalse(registry._verify_downloaded_file(test_file, len(content) + 100))

        # Test with nonexistent file
        self.assertFalse(registry._verify_downloaded_file(self.test_dir / "nonexistent", 2048))

    def test_get_model_with_cached_file(self):
        """Test get_model with an existing cached file."""
        # Create a mock file in the cache
        cached_file = self.test_dir / "model.pt"
        cached_file.write_text("x" * 2048)  # Large enough to pass verification

        # Create a mock registry that tracks if download was called
        class TrackingRegistry(self.MockModelRegistry):
            download_called = False

            def _download_to_file(self, filename: str) -> str:
                self.download_called = True
                return "mock_download_path"

        registry = TrackingRegistry(
            model_url="mock://valid/model.pt", model_cache_dir=self.test_dir, api_key="test_key"
        )

        # Test with existing valid file
        result = registry.get_model()
        self.assertEqual(result, str(cached_file))  # Should use cached file
        self.assertFalse(registry.download_called)

        # Test with invalid file (too small)
        cached_file.write_text("x" * 512)  # Too small to pass verification

        registry.download_called = False
        result = registry.get_model()
        self.assertEqual(result, "mock_download_path")  # Should download new file
        self.assertTrue(registry.download_called)

    def test_get_model_fresh_download(self):
        """Test get_model with a fresh download."""

        # Setup registry with a mock download method
        class DownloadingRegistry(self.MockModelRegistry):
            def _download_to_file(self, filename: str) -> str:
                return str(self.model_cache_dir / filename)

        registry = DownloadingRegistry(
            model_url="mock://valid/model.pt", model_cache_dir=self.test_dir, api_key="test_key"
        )

        # Test a fresh download
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.is_dir", return_value=True),
        ):
            result = registry.get_model()  # Should call _download_to_file
            expected_file = self.test_dir / "model.pt"
            self.assertEqual(result, str(expected_file))

    def test_get_credential_from_netrc_success(self):
        """Test successful credential retrieval from netrc."""
        # Mock netrc
        with patch("netrc.netrc") as mock_netrc_class:
            mock_netrc = MagicMock()
            mock_netrc_class.return_value = mock_netrc
            mock_netrc.authenticators.return_value = ("user", None, "test_credential")

            # Test retrieval
            credential = ModelRegistry._get_credential_from_netrc("test.domain.com")
            self.assertEqual(credential, "test_credential")
            mock_netrc.authenticators.assert_called_once_with("test.domain.com")

    def test_get_credential_from_netrc_missing(self):
        """Test credential retrieval when netrc file is missing."""
        # Mock netrc file not found
        with patch("netrc.netrc", side_effect=FileNotFoundError("No .netrc file")):
            with self.assertRaises(ModelRegistryError) as cm:
                ModelRegistry._get_credential_from_netrc("test.domain.com")
            self.assertIn("Failed to read .netrc file", str(cm.exception))

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
            registry = NgcModelRegistry(
                model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                model_cache_dir=self.test_dir,
                api_key="",
            )
            with self.assertRaises(ModelRegistryError) as cm:
                registry._download_to_file("test_file.pt")
            self.assertIn("API key cannot be empty", str(cm.exception))


class TestNgcModelRegistry(unittest.TestCase):
    """Test cases for the NgcModelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Mock requests Session
        self.session_patcher = patch("requests.Session")
        self.mock_session_class = self.session_patcher.start()
        self.mock_session = MagicMock()
        self.mock_session_class.return_value = self.mock_session
        self.addCleanup(self.session_patcher.stop)

        # Mock netrc
        self.netrc_patcher = patch("netrc.netrc")
        self.mock_netrc_class = self.netrc_patcher.start()
        self.mock_netrc = MagicMock()
        self.mock_netrc_class.return_value = self.mock_netrc
        self.addCleanup(self.netrc_patcher.stop)

        self.registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="nvapi-test_api_key",  # non-legacy API key
        )

        self.registry_legacy = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",  # legacy API key
        )

    def tearDown(self):
        """Clean up test artifacts."""
        self.temp_dir.cleanup()

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        self.assertEqual(self.registry.model_url, "https://api.ngc.nvidia.com/v2/models/test/model")
        self.assertEqual(self.registry.api_key, "nvapi-test_api_key")
        # Session should not be initialized yet
        self.assertIsNone(self.registry.session)
        # Verify session headers are not set yet
        self.mock_session.headers.update.assert_not_called()

    def test_lazy_session_initialization(self):
        """Test that session is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="nvapi-test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify session was created with correct headers
                    mock_session.return_value.headers.update.assert_called_once_with(
                        {
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                        }
                    )
                    self.assertEqual(mock_session.return_value.auth.token, "nvapi-test_api_key")

    def test_lazy_api_key_initialization(self):
        """Test that API key is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="nvapi-test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify API key was set
                    self.assertEqual(registry.api_key, "nvapi-test_api_key")

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="",
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._download_to_file("test_file.pt")
        self.assertIn("API key cannot be empty", str(cm.exception))

    def test_validate_url_valid(self):
        """Test URL validation with valid NGC URL."""
        self.assertTrue(self.registry._validate_url("https://api.ngc.nvidia.com/v2/models/test/model"))

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URL."""
        self.assertFalse(self.registry._validate_url("https://invalid-url.com"))

    def test_domain_property(self):
        """Test the domain property returns the correct value."""
        self.assertEqual(self.registry.domain, "api.ngc.nvidia.com")

    def test_get_api_key_from_env_var(self):
        """Test API key retrieval from NGC_API_KEY environment variable."""

        # Mock os.environ.get to return a test API key only for NGC_API_KEY
        def mock_get_env(key, default=None):
            if key == "NGC_API_KEY":
                return "nvapi-env_api_key"
            return default

        # Create a new registry without an API key
        with patch("os.environ.get", side_effect=mock_get_env):
            # Also mock netrc to ensure it's not used
            with patch.object(ModelRegistry, "_get_credential_from_netrc", return_value=None):
                registry = NgcModelRegistry(
                    model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                    model_cache_dir=self.test_dir,
                    api_key=None,
                )

                self.assertEqual(registry._get_api_key(), "nvapi-env_api_key")

    def test_api_key_precedence(self):
        """Test that API key precedence is correctly followed: direct param > env var > netrc."""

        # Set up mocks for all three methods
        def mock_get_env(key, default=None):
            if key == "NGC_API_KEY":
                return "nvapi-env_api_key"
            return default

        # Test direct param takes precedence
        with patch("os.environ.get", side_effect=mock_get_env):
            with patch.object(ModelRegistry, "_get_credential_from_netrc", return_value="nvapi-netrc_api_key"):
                registry = NgcModelRegistry(
                    model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                    model_cache_dir=self.test_dir,
                    api_key="nvapi-direct_api_key",
                )
                # Force API key initialization by calling _download_to_file
                with patch("requests.Session") as mock_session:
                    mock_response = MagicMock()
                    mock_response.headers = {"content-length": "100"}
                    mock_response.iter_content.return_value = [b"x" * 100]
                    mock_response.raise_for_status.return_value = None
                    mock_session.return_value.get.return_value = mock_response
                    with (
                        patch("builtins.open", mock_open()),
                        patch.object(registry, "_verify_downloaded_file", return_value=True),
                    ):
                        registry._download_to_file("test_file.pt")
                        self.assertEqual(registry.api_key, "nvapi-direct_api_key")

        # Test env var takes precedence over netrc
        with patch("os.environ.get", side_effect=mock_get_env):
            with patch.object(ModelRegistry, "_get_credential_from_netrc", return_value="nvapi-netrc_api_key"):
                registry = NgcModelRegistry(
                    model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                    model_cache_dir=self.test_dir,
                    api_key=None,
                )
                # Force API key initialization by calling _download_to_file
                with patch("requests.Session") as mock_session:
                    mock_response = MagicMock()
                    mock_response.headers = {"content-length": "100"}
                    mock_response.iter_content.return_value = [b"x" * 100]
                    mock_response.raise_for_status.return_value = None
                    mock_session.return_value.get.return_value = mock_response
                    with (
                        patch("builtins.open", mock_open()),
                        patch.object(registry, "_verify_downloaded_file", return_value=True),
                    ):
                        registry._download_to_file("test_file.pt")
                        self.assertEqual(registry.api_key, "nvapi-env_api_key")

    def test_extract_org_team_from_model_url_valid(self):
        """Test extraction of org and team from model URL."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )
        self.assertEqual(registry._extract_org_team_from_model_url(), ("my_org", "my_team"))

    def test_extract_org_team_from_model_url_invalid(self):
        """Test extraction of org and team from model URL."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._extract_org_team_from_model_url()
        self.assertIn(
            "Invalid NGC URL format, cannot extract org/team: https://api.ngc.nvidia.com/v2/models/test/model",
            str(cm.exception),
        )

    def test_request_session_token_from_legacy_api_key_success(self):
        """Test successful session token request from legacy API key."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call
        with patch("requests.get") as mock_get:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"token": "my_session_token"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test the method
            result = registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            # Verify the result
            self.assertEqual(result, "my_session_token")

            # Verify the request was made correctly
            mock_get.assert_called_once_with(
                "https://authn.nvidia.com/token?service=ngc&scope=group/ngc:my_org&group/ngc:my_org/my_team",
                auth=("$oauthtoken", "legacy_api_key"),
                headers={"Accept": "application/json"},
                timeout=30,
            )

    def test_request_session_token_from_legacy_api_key_no_token_in_response(self):
        """Test session token request when response doesn't contain token."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call with response missing token
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"error": "invalid_request"}  # No "token" field
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test that it raises ModelRegistryError
            with self.assertRaises(ModelRegistryError) as cm:
                registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            self.assertIn("No token found in authentication response", str(cm.exception))

    def test_request_session_token_from_legacy_api_key_request_failure(self):
        """Test session token request when HTTP request fails."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call to raise an exception
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Connection error")

            # Test that it raises ModelRegistryError
            with self.assertRaises(ModelRegistryError) as cm:
                registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            self.assertIn("Failed to exchange legacy API key for session token", str(cm.exception))

    def test_request_session_token_from_legacy_api_key_invalid_json(self):
        """Test session token request when response has invalid JSON."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call with invalid JSON response
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test that it raises ModelRegistryError
            with self.assertRaises(ModelRegistryError) as cm:
                registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            self.assertIn("Invalid response format from NGC authentication service", str(cm.exception))

    def test_request_session_token_from_api_key_legacy_token(self):
        """Test _request_session_token_from_api_key with legacy API key (non-PAT)."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call that happens in _request_session_token_from_legacy_api_key
        with patch("requests.get") as mock_get:
            # Setup mock response for the token exchange
            mock_response = MagicMock()
            mock_response.json.return_value = {"token": "exchanged_session_token"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test with legacy API key (doesn't start with "nvapi-")
            result = registry._request_session_token_from_api_key("legacy_api_key")

            # Verify the HTTP request was made correctly
            mock_get.assert_called_once_with(
                "https://authn.nvidia.com/token?service=ngc&scope=group/ngc:my_org&group/ngc:my_org/my_team",
                auth=("$oauthtoken", "legacy_api_key"),
                headers={"Accept": "application/json"},
                timeout=30,
            )

            # Verify the result
            self.assertEqual(result, "exchanged_session_token")


class TestGitLabModelRegistry(unittest.TestCase):
    """Test cases for the GitLabModelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Mock requests Session
        self.session_patcher = patch("requests.Session")
        self.mock_session_class = self.session_patcher.start()
        self.mock_session = MagicMock()
        self.mock_session_class.return_value = self.mock_session
        self.addCleanup(self.session_patcher.stop)

        # Mock netrc
        self.netrc_patcher = patch("netrc.netrc")
        self.mock_netrc_class = self.netrc_patcher.start()
        self.mock_netrc = MagicMock()
        self.mock_netrc_class.return_value = self.mock_netrc
        self.addCleanup(self.netrc_patcher.stop)

        self.registry = GitLabModelRegistry(
            model_url=_GITLAB_MODEL_URL,
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
            domain=_GITLAB_TEST_DOMAIN,
        )

    def tearDown(self):
        """Clean up test artifacts."""
        self.temp_dir.cleanup()

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        self.assertEqual(self.registry.model_url, _GITLAB_MODEL_URL)
        self.assertEqual(self.registry.api_key, "test_api_key")
        # Session should not be initialized yet
        self.assertIsNone(self.registry.session)
        # Verify session headers are not set yet
        self.mock_session.headers.update.assert_not_called()

    def test_lazy_session_initialization(self):
        """Test that session is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = GitLabModelRegistry(
            model_url=_GITLAB_MODEL_URL,
            model_cache_dir=self.test_dir,
            domain=_GITLAB_TEST_DOMAIN,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify session was created with correct headers
                    mock_session.return_value.headers.update.assert_called_once_with(
                        {
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                        }
                    )
                    self.assertEqual(mock_session.return_value.auth.token, "test_api_key")

    def test_lazy_api_key_initialization(self):
        """Test that API key is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = GitLabModelRegistry(
            model_url=_GITLAB_MODEL_URL,
            model_cache_dir=self.test_dir,
            domain=_GITLAB_TEST_DOMAIN,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify API key was set
                    self.assertEqual(registry.api_key, "test_api_key")

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        registry = GitLabModelRegistry(
            model_url=_GITLAB_MODEL_URL,
            model_cache_dir=self.test_dir,
            domain=_GITLAB_TEST_DOMAIN,
            api_key="",
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._download_to_file("test_file.pt")
        self.assertIn("API key cannot be empty", str(cm.exception))

    def test_validate_url_valid(self):
        """Test URL validation with valid GitLab URL."""
        self.assertTrue(self.registry._validate_url(_GITLAB_MODEL_URL))

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URL."""
        self.assertFalse(self.registry._validate_url("https://invalid-url.com"))

    def test_domain_property(self):
        """Test the domain property returns the correct value."""
        self.assertEqual(self.registry.domain, _GITLAB_TEST_DOMAIN)

    def test_default_domain(self):
        """Test the domain defaults to gitlab.com."""
        registry = GitLabModelRegistry(
            model_url="https://gitlab.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )
        self.assertEqual(GitLabModelRegistry.DEFAULT_GITLAB_DOMAIN, "gitlab.com")
        self.assertEqual(registry.domain, "gitlab.com")

    def test_url_for_other_domain_rejected(self):
        """Test a URL on a different host than the configured domain is rejected."""
        with self.assertRaises(ModelRegistryError):
            GitLabModelRegistry(
                model_url="https://gitlab.com/projects/models/test/model.tar.gz",
                model_cache_dir=self.test_dir,
                domain=_GITLAB_TEST_DOMAIN,
            )

    def test_api_key_hint_uses_instance_domain(self):
        """Test the API key hint points at the configured GitLab instance."""
        self.assertIn(
            f"https://{_GITLAB_TEST_DOMAIN}/-/user_settings/personal_access_tokens", self.registry.api_key_hint()
        )


class TestHuggingFaceModelRegistry(unittest.TestCase):
    """Test cases for the Hugging Face model registry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.model_url = "https://huggingface.co/nvidia/Fixer/resolve/main/pretrained/pretrained_fixer.pkl"
        self.registry = HuggingFaceModelRegistry(
            model_url=self.model_url,
            model_cache_dir=self.test_dir,
            api_key="hf-direct-token",
        )

    def tearDown(self):
        """Clean up test artifacts."""
        self.temp_dir.cleanup()

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        self.assertEqual(self.registry.model_url, self.model_url)
        self.assertEqual(self.registry.api_key, "hf-direct-token")

    def test_parse_resolve_url(self):
        """Test parsing a Hugging Face resolve URL."""
        repo_id, revision, filename = self.registry._parse_model_url()
        self.assertEqual(repo_id, "nvidia/Fixer")
        self.assertEqual(revision, "main")
        self.assertEqual(filename, "pretrained/pretrained_fixer.pkl")

    def test_parse_blob_url(self):
        """Test parsing a Hugging Face blob URL."""
        registry = HuggingFaceModelRegistry(
            model_url="https://huggingface.co/nvidia/Fixer/blob/main/base/model_fast_tokenizer.pt",
            model_cache_dir=self.test_dir,
            api_key=None,
        )
        repo_id, revision, filename = registry._parse_model_url()
        self.assertEqual(repo_id, "nvidia/Fixer")
        self.assertEqual(revision, "main")
        self.assertEqual(filename, "base/model_fast_tokenizer.pt")

    def test_parse_invalid_url(self):
        """Test parsing a malformed Hugging Face URL."""
        registry = HuggingFaceModelRegistry(
            model_url="https://huggingface.co/nvidia/Fixer",
            model_cache_dir=self.test_dir,
            api_key=None,
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._parse_model_url()
        self.assertIn("Invalid Hugging Face URL format", str(cm.exception))

    def test_get_api_key_from_env_var(self):
        """Test Hugging Face token retrieval from HF_TOKEN."""

        def mock_get_env(key, default=None):
            if key == "HF_TOKEN":
                return "hf-env-token"
            return default

        with patch("os.environ.get", side_effect=mock_get_env):
            registry = HuggingFaceModelRegistry(
                model_url=self.model_url,
                model_cache_dir=self.test_dir,
                api_key=None,
            )
            self.assertEqual(registry._get_api_key(), "hf-env-token")

    def test_public_download_allows_missing_api_key(self):
        """Test that no token is required for public Hugging Face repositories."""
        with patch.dict("os.environ", {}, clear=True):
            registry = HuggingFaceModelRegistry(
                model_url=self.model_url,
                model_cache_dir=self.test_dir,
                api_key=None,
            )
            self.assertIsNone(registry._get_api_key())

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        registry = HuggingFaceModelRegistry(
            model_url=self.model_url,
            model_cache_dir=self.test_dir,
            api_key="",
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._download_to_file("pretrained_fixer.pkl")
        self.assertIn("API key cannot be empty", str(cm.exception))

    @patch("huggingface_hub.hf_hub_download")
    def test_download_to_file_uses_hf_hub_download(self, mock_hf_hub_download):
        """Test that Hugging Face downloads are delegated to hf_hub_download."""
        cached_file = self.test_dir / "pretrained_fixer.pkl"
        downloaded_file = self.test_dir / "pretrained" / "pretrained_fixer.pkl"
        downloaded_file.parent.mkdir(parents=True)
        downloaded_file.write_text("x" * 2048)
        mock_hf_hub_download.return_value = str(downloaded_file)

        result = self.registry._download_to_file("pretrained_fixer.pkl")

        self.assertEqual(result, str(cached_file))
        self.assertFalse(downloaded_file.exists())
        mock_hf_hub_download.assert_called_once_with(
            repo_id="nvidia/Fixer",
            filename="pretrained/pretrained_fixer.pkl",
            revision="main",
            local_dir=self.test_dir,
            token="hf-direct-token",
        )

    @patch("huggingface_hub.hf_hub_download")
    def test_download_uses_hf_token_when_direct_api_key_missing(self, mock_hf_hub_download):
        """Test that HF_TOKEN is used when no direct API key is provided."""
        downloaded_file = self.test_dir / "pretrained" / "pretrained_fixer.pkl"
        downloaded_file.parent.mkdir(parents=True)
        downloaded_file.write_text("x" * 2048)
        mock_hf_hub_download.return_value = str(downloaded_file)

        with patch.dict("os.environ", {"HF_TOKEN": "hf-env-token"}, clear=True):
            registry = HuggingFaceModelRegistry(
                model_url=self.model_url,
                model_cache_dir=self.test_dir,
                api_key=None,
            )
            registry._download_to_file("pretrained_fixer.pkl")

        mock_hf_hub_download.assert_called_once_with(
            repo_id="nvidia/Fixer",
            filename="pretrained/pretrained_fixer.pkl",
            revision="main",
            local_dir=self.test_dir,
            token="hf-env-token",
        )

    @patch("huggingface_hub.hf_hub_download")
    def test_download_allows_no_token(self, mock_hf_hub_download):
        """Test public Hugging Face downloads pass token=None."""
        downloaded_file = self.test_dir / "pretrained" / "pretrained_fixer.pkl"
        downloaded_file.parent.mkdir(parents=True)
        downloaded_file.write_text("x" * 2048)
        mock_hf_hub_download.return_value = str(downloaded_file)

        with patch.dict("os.environ", {}, clear=True):
            registry = HuggingFaceModelRegistry(
                model_url=self.model_url,
                model_cache_dir=self.test_dir,
                api_key=None,
            )
            registry._download_to_file("pretrained_fixer.pkl")

        mock_hf_hub_download.assert_called_once_with(
            repo_id="nvidia/Fixer",
            filename="pretrained/pretrained_fixer.pkl",
            revision="main",
            local_dir=self.test_dir,
            token=None,
        )

    def test_parse_tree_url_fails_with_clear_error(self):
        """Test tree URLs fail before trying to download a directory."""
        registry = HuggingFaceModelRegistry(
            model_url="https://huggingface.co/nvidia/Fixer/tree/main/pretrained",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        with self.assertRaises(ModelRegistryError) as cm:
            registry._parse_model_url()

        self.assertIn("Unsupported Hugging Face URL type 'tree'", str(cm.exception))

    def test_download_fails_when_huggingface_hub_missing(self):
        """Test a clear error is raised if the optional Hugging Face dependency is missing."""
        real_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                raise ImportError("missing huggingface_hub")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ModelRegistryError) as cm:
                self.registry._download_to_file("pretrained_fixer.pkl")

        self.assertIn("huggingface-hub is required for Hugging Face model URLs", str(cm.exception))

    def test_get_model_with_cached_file(self):
        """Test get_model returns an existing Hugging Face cache file."""
        cached_file = self.test_dir / "pretrained_fixer.pkl"
        cached_file.write_text("x" * 2048)

        with patch("huggingface_hub.hf_hub_download") as mock_hf_hub_download:
            result = self.registry.get_model()

        self.assertEqual(result, str(cached_file))
        mock_hf_hub_download.assert_not_called()

    @patch("huggingface_hub.hf_hub_download")
    def test_get_model_reuses_huggingface_download_cache(self, mock_hf_hub_download):
        """Test get_model reuses the flat cache file populated by a Hugging Face download."""
        expected_cached_file = self.test_dir / "pretrained_fixer.pkl"

        def fake_hf_hub_download(**kwargs):
            downloaded_file = Path(kwargs["local_dir"]) / kwargs["filename"]
            downloaded_file.parent.mkdir(parents=True, exist_ok=True)
            downloaded_file.write_text("x" * 2048)
            return str(downloaded_file)

        mock_hf_hub_download.side_effect = fake_hf_hub_download

        first_result = self.registry.get_model()
        second_result = self.registry.get_model()

        self.assertEqual(first_result, str(expected_cached_file))
        self.assertEqual(second_result, str(expected_cached_file))
        self.assertTrue(expected_cached_file.exists())
        mock_hf_hub_download.assert_called_once()

    def test_domain_property(self):
        """Test the domain property returns the correct value."""
        self.assertEqual(self.registry.domain, "huggingface.co")


class TestFetchBundle(unittest.TestCase):
    """Test cases for the fetch_bundle multi-file Hugging Face bundle helper."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test artifacts."""
        self.temp_dir.cleanup()

    @patch("huggingface_hub.snapshot_download")
    def test_fetch_bundle_parses_manifest(self, mock_snapshot_download):
        """Test a successful fetch returns the snapshot dir and its parsed config.json."""
        snapshot_dir = self.test_dir / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "config.json").write_text('{"components": []}')
        mock_snapshot_download.return_value = str(snapshot_dir)

        result = fetch_bundle(
            repo_id="nvidia/asset-harvester",
            revision="main",
            cache_dir=str(self.test_dir / "cache"),
            allow_patterns=["config.json"],
        )

        self.assertEqual(result.path, snapshot_dir)
        self.assertEqual(result.manifest, {"components": []})
        mock_snapshot_download.assert_called_once_with(
            repo_id="nvidia/asset-harvester",
            revision="main",
            cache_dir=str(self.test_dir / "cache"),
            allow_patterns=["config.json"],
        )

    @patch("huggingface_hub.snapshot_download")
    def test_fetch_bundle_missing_manifest_is_none(self, mock_snapshot_download):
        """Test a snapshot with no config.json returns a None manifest instead of failing."""
        snapshot_dir = self.test_dir / "snapshot"
        snapshot_dir.mkdir()
        mock_snapshot_download.return_value = str(snapshot_dir)

        result = fetch_bundle(repo_id="nvidia/asset-harvester", revision="main", cache_dir=str(self.test_dir))

        self.assertIsNone(result.manifest)

    @patch("huggingface_hub.snapshot_download")
    def test_fetch_bundle_translates_hub_error(self, mock_snapshot_download):
        """Test a Hugging Face Hub download error becomes a ModelRegistryError."""
        from huggingface_hub.errors import HfHubHTTPError

        mock_snapshot_download.side_effect = HfHubHTTPError("boom", response=MagicMock())

        with self.assertRaises(ModelRegistryError):
            fetch_bundle(repo_id="nvidia/asset-harvester", revision="main", cache_dir=str(self.test_dir))

    @patch("huggingface_hub.snapshot_download")
    def test_fetch_bundle_translates_malformed_manifest(self, mock_snapshot_download):
        """Test a snapshot with an unparseable config.json becomes a ModelRegistryError."""
        snapshot_dir = self.test_dir / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "config.json").write_text("not valid json")
        mock_snapshot_download.return_value = str(snapshot_dir)

        with self.assertRaises(ModelRegistryError):
            fetch_bundle(repo_id="nvidia/asset-harvester", revision="main", cache_dir=str(self.test_dir))

    def test_fetch_bundle_fails_when_huggingface_hub_missing(self):
        """Test a clear error is raised if the optional Hugging Face dependency is missing."""
        real_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                raise ImportError("missing huggingface_hub")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ModelRegistryError) as cm:
                fetch_bundle(repo_id="nvidia/asset-harvester", revision="main", cache_dir=str(self.test_dir))

        self.assertIn("huggingface-hub is required", str(cm.exception))


class TestModelRegistryFactory(unittest.TestCase):
    """Test cases for the model registry factory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        # Isolate registered GitLab domains and the env var from other tests.
        domains_patcher = patch("nurec.utils.model_registry._REGISTERED_GITLAB_DOMAINS", set())
        domains_patcher.start()
        self.addCleanup(domains_patcher.stop)
        env_patcher = patch.dict(os.environ)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        os.environ.pop("NUREC_GITLAB_DOMAINS", None)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_create_ngc_registry(self):
        """Test creation of NgcModelRegistry from factory."""
        registry = create_model_registry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )
        self.assertIsInstance(registry, NgcModelRegistry)
        self.assertEqual(registry.model_url, "https://api.ngc.nvidia.com/v2/models/test/model")
        self.assertEqual(registry.api_key, "test_api_key")

    def test_create_gitlab_registry(self):
        """Test creation of GitLabModelRegistry from factory for a registered domain."""
        register_gitlab_domain(_GITLAB_TEST_DOMAIN)
        registry = create_model_registry(
            model_url=_GITLAB_MODEL_URL,
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )
        self.assertIsInstance(registry, GitLabModelRegistry)
        self.assertEqual(registry.model_url, _GITLAB_MODEL_URL)
        self.assertEqual(registry.api_key, "test_api_key")
        self.assertEqual(registry.domain, _GITLAB_TEST_DOMAIN)

    def test_create_gitlab_registry_default_domain(self):
        """Test gitlab.com is dispatched to GitLabModelRegistry without registration."""
        registry = create_model_registry(
            model_url="https://gitlab.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )
        self.assertIsInstance(registry, GitLabModelRegistry)
        self.assertEqual(registry.domain, "gitlab.com")

    def test_create_gitlab_registry_from_env_var(self):
        """Test NUREC_GITLAB_DOMAINS enables domains, ignoring whitespace and blank entries."""
        os.environ["NUREC_GITLAB_DOMAINS"] = f" , other.example.com ,, {_GITLAB_TEST_DOMAIN} ,"
        registry = create_model_registry(
            model_url=_GITLAB_MODEL_URL, model_cache_dir=self.test_dir, api_key="test_api_key"
        )
        self.assertIsInstance(registry, GitLabModelRegistry)
        self.assertEqual(registry.domain, _GITLAB_TEST_DOMAIN)
        other = create_model_registry(
            model_url="https://other.example.com/a/b", model_cache_dir=self.test_dir, api_key="test_api_key"
        )
        self.assertIsInstance(other, GitLabModelRegistry)
        self.assertEqual(other.domain, "other.example.com")

    def test_env_var_blank_does_not_enable_domains(self):
        """Test an empty NUREC_GITLAB_DOMAINS does not enable any extra domain."""
        os.environ["NUREC_GITLAB_DOMAINS"] = " , ,"
        with self.assertRaises(ModelRegistryError):
            create_model_registry(model_url=_GITLAB_MODEL_URL, model_cache_dir=self.test_dir, api_key="test_api_key")

    def test_unregistered_gitlab_domain_rejected(self):
        """Test a GitLab-like host that is not registered is unsupported."""
        with self.assertRaises(ModelRegistryError):
            create_model_registry(model_url=_GITLAB_MODEL_URL, model_cache_dir=self.test_dir, api_key="test_api_key")

    def test_register_gitlab_domain_normalizes_and_is_idempotent(self):
        """Test registration strips/lowercases the domain and can be repeated."""
        register_gitlab_domain(f"  {_GITLAB_TEST_DOMAIN.upper()} ")
        register_gitlab_domain(_GITLAB_TEST_DOMAIN)
        registry = create_model_registry(
            model_url=_GITLAB_MODEL_URL, model_cache_dir=self.test_dir, api_key="test_api_key"
        )
        self.assertIsInstance(registry, GitLabModelRegistry)

    def test_register_gitlab_domain_rejects_invalid(self):
        """Test invalid domains are rejected."""
        for bad in ("", "   ", f"https://{_GITLAB_TEST_DOMAIN}", f"{_GITLAB_TEST_DOMAIN}/path", "bad host.com"):
            with self.subTest(domain=bad), self.assertRaises(ValueError):
                register_gitlab_domain(bad)

    def test_create_huggingface_registry(self):
        """Test creation of HuggingFaceModelRegistry from factory."""
        registry = create_model_registry(
            model_url="https://huggingface.co/nvidia/Fixer/resolve/main/pretrained/pretrained_fixer.pkl",
            model_cache_dir=self.test_dir,
            api_key="hf-test-token",
        )
        self.assertIsInstance(registry, HuggingFaceModelRegistry)
        self.assertEqual(
            registry.model_url,
            "https://huggingface.co/nvidia/Fixer/resolve/main/pretrained/pretrained_fixer.pkl",
        )
        self.assertEqual(registry.api_key, "hf-test-token")

    def test_create_registry_rejects_plaintext_http(self):
        """Test that authenticated registries reject plaintext URLs."""
        register_gitlab_domain(_GITLAB_TEST_DOMAIN)
        for model_url in (
            "http://api.ngc.nvidia.com/v2/models/test/model",
            _GITLAB_MODEL_URL_HTTP,
        ):
            with self.subTest(model_url=model_url), self.assertRaises(ModelRegistryError):
                create_model_registry(
                    model_url=model_url,
                    model_cache_dir=self.test_dir,
                    api_key="test_api_key",
                )

    def test_unsupported_domain(self):
        """Test factory with unsupported domain."""
        with self.assertRaises(ModelRegistryError) as cm:
            create_model_registry(
                model_url="https://unsupported-domain.com/model",
                model_cache_dir=self.test_dir,
                api_key="test_api_key",
            )
        self.assertIn("Unsupported model registry domain", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
