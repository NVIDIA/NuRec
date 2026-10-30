# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for downloading models from various model registries.

GitLab hosts other than gitlab.com can be enabled for `create_model_registry` by calling
`register_gitlab_domain` or by setting NUREC_GITLAB_DOMAINS to a comma-separated list of hosts.
"""

import json
import logging
import netrc
import os
import re
import shutil

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, NoReturn, Optional, Type, TypeVar
from urllib.parse import unquote, urlparse

import requests

from requests.auth import AuthBase
from tqdm import tqdm

from nurec.utils.misc import unpack_optional


logger = logging.getLogger(__name__)

ExceptionType = TypeVar("ExceptionType", bound=Exception)


def partial_api_key(api_key: str, clear_view_length: int) -> str:
    """Helper function to partially mask an API key for logging.

    Args:
        api_key: The API key to mask.
        clear_view_length: Length of the API key to show unmasked.

    WARNING: if the API key is shorter than or equal to clear_view_length,
             the full API key will be returned unmasked.

    Returns:
        str: The partially masked API key.
    """

    if len(api_key) <= clear_view_length:
        return api_key

    return api_key[:clear_view_length] + "*" * (len(api_key) - clear_view_length)


def log_and_raise(exception: Type[ExceptionType], message: str, *args, **kwargs) -> NoReturn:
    """Helper function to log an error and raise an exception.

    Args:
        exception: Exception class to raise
        message: Error message to log and include in exception
        *args: Format args for the message
        **kwargs: Additional kwargs for exception

    Raises:
        ExceptionType: The specified exception type
    """
    formatted_message = message % args if args else message
    logger.error(formatted_message)
    raise exception(formatted_message, **kwargs)


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""

    pass


class ModelRegistry(ABC):
    """Abstract base class for model registry implementations.

    This class defines the interface for downloading models from different model
    registries. Each specific registry implementation must inherit from this class
    and implement the download method.

    Attributes:
        model_url (str): URL where the model can be downloaded from.
        model_cache_dir (Path): Directory where downloaded models should be cached.
        api_key (str): API key for authenticating with the registry. If not provided,
            it will be retrieved from the .netrc file.
    """

    MIN_FILE_SIZE = 1024  # 1KB minimum file size
    CHUNK_SIZE = 8192  # 8KB chunks for downloading

    def __init__(self, model_url: str, model_cache_dir: Path, api_key: Optional[str] = None) -> None:
        """Initialize the model registry.

        Args:
            model_url: URL where the model can be downloaded from.
            model_cache_dir: Directory where downloaded models should be cached.

        Raises:
            ModelRegistryError: If the model URL is invalid.
        """
        logger.debug("Initializing model registry with URL: %s and cache dir: %s", model_url, model_cache_dir)
        if not self._validate_url(model_url):
            log_and_raise(ModelRegistryError, "Invalid model URL: %s", model_url)
        self.model_url = model_url
        self.model_cache_dir = model_cache_dir
        logger.debug("Model registry initialized successfully")
        self.api_key = api_key
        self.session: Optional[requests.Session] = None
        super().__init__()

    def _get_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """Get the API key for the model registry.

        The API key is resolved in the following order of precedence:
        1. Directly provided api_key parameter
        2. .netrc file credentials for the registry domain

        Args:
            api_key: Optional API key provided directly.

        Returns:
            Optional[str]: The API key for the model registry.

        Raises:
            ModelRegistryError: If no API key could be found or if an empty API key was provided.
        """
        # Validate API key if provided
        if api_key is not None and not api_key.strip():
            log_and_raise(ModelRegistryError, "API key cannot be empty")

        # Get API key from netrc if not provided
        if not api_key:
            logger.debug("No API key provided, attempting to get it from the .netrc file.")
            try:
                api_key = ModelRegistry._get_credential_from_netrc(self.domain)
                logger.debug("Successfully retrieved API key from the .netrc file.")
            except Exception as e:
                log_and_raise(ModelRegistryError, "Failed to get API key from .netrc file: %s", str(e))

        if not api_key:
            log_and_raise(ModelRegistryError, "API key must be provided either directly or via .netrc file")

        return api_key

    def _get_session(self, session_token: str) -> requests.Session:
        """Get a session for the model registry.

        Returns:
            requests.Session: A session for the model registry.
        """

        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        class BearerAuth(AuthBase):
            """Custom authentication handler for OAuth2 Bearer tokens.

            Args:
                token: The OAuth2 bearer token to use for authentication
            """

            def __init__(self, token):
                if not token or not token.strip():
                    raise ValueError("Token cannot be empty")

                self.token = token

            def __call__(self, r):
                r.headers["Authorization"] = f"Bearer {self.token}"
                return r

        session.auth = BearerAuth(session_token)

        logger.debug("Session initialized with authentication headers")

        return session

    def _request_session_token_from_api_key(self, api_key: str) -> str:
        """The session token is the API key in most cases, e.g. for NGCModelRegistry this could be different if a legacy access token is used.

        Args:
            api_key: The API key to use for authentication.

        Returns:
            str: A session token for the model registry.
        """
        if not api_key:
            log_and_raise(ModelRegistryError, "API key cannot be empty")

        return api_key

    @property
    @abstractmethod
    def domain(self) -> str:
        """Get the domain of the model registry.

        Returns:
            str: The domain of the model registry.
        """
        pass

    @abstractmethod
    def api_key_hint(self) -> str:
        """Get backend-dependent hint for the API source / key format.

        Returns:
            str: A hint for the API source / key format.
        """
        pass

    def _validate_url(self, url: str) -> bool:
        """Validate that an HTTPS URL belongs to this registry.

        Args:
            url: The URL to validate.

        Returns:
            bool: True if the URL is valid for this registry.
        """
        logger.debug("Validating URL: %s", url)
        try:
            parsed = urlparse(url)
            is_valid = parsed.scheme.lower() == "https" and parsed.netloc == self.domain
            if not is_valid:
                logger.warning(
                    "Invalid model URL scheme or domain: %s://%s (expected https://%s)",
                    parsed.scheme,
                    parsed.netloc,
                    self.domain,
                )
            return is_valid
        except Exception as e:
            logger.warning("Failed to parse URL: %s - %s", url, str(e))
            return False

    def _verify_cached_file(self, file_path: Path) -> bool:
        """Verify if a cached file meets minimum size requirements.

        Args:
            file_path: Path to the cached file.

        Returns:
            bool: True if the file meets minimum size requirements.
        """
        logger.debug("Verifying cached file: %s", file_path)

        try:
            actual_size = file_path.stat().st_size
        except OSError as e:
            logger.warning("Failed to get file size: %s", str(e))
            return False

        if actual_size < self.MIN_FILE_SIZE:
            logger.warning(
                "Cached file size (%d bytes) is below minimum required size (%d bytes)", actual_size, self.MIN_FILE_SIZE
            )
            return False

        return True

    def _verify_downloaded_file(self, file_path: Path, expected_size: int) -> bool:
        """Verify if a newly downloaded file matches the expected size.

        Args:
            file_path: Path to the downloaded file.
            expected_size: Expected file size from Content-Length header.

        Returns:
            bool: True if the file size matches expected size.
        """
        logger.debug("Verifying downloaded file: %s", file_path)

        try:
            actual_size = file_path.stat().st_size
        except OSError as e:
            logger.warning("Failed to get file size: %s", str(e))
            return False

        if actual_size != expected_size:
            logger.warning("Downloaded file size mismatch: expected %d bytes, got %d bytes", expected_size, actual_size)
            return False

        return True

    def _download_to_file(self, filename: str) -> str:
        """Download a file from the registry to the cache.

        This method handles the complete download process, including:
        - Preparing the download request using the registry-specific implementation
        - Creating necessary directories
        - Performing the actual download with progress tracking
        - Verifying the downloaded file

        Args:
            filename: The target filename for the downloaded file

        Returns:
            str: Path to the downloaded file

        Raises:
            ModelRegistryError: If the download fails
        """
        logger.info(f"Starting model download from {self.__class__.__name__}")

        # Create cache directory if it doesn't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create the full path to the cached file
        cached_file = self.model_cache_dir / filename

        try:
            # Lazy initialization of API key
            if self.api_key is None:
                self.api_key = self._get_api_key()
            if not self.api_key:
                log_and_raise(ModelRegistryError, "API key cannot be empty")

            session_token = self._request_session_token_from_api_key(self.api_key)

            # Lazy initialization of session
            if self.session is None:
                self.session = self._get_session(session_token)

            logger.info("Downloading model from %s", self.model_url)  # Log URL without credentials
            response = self.session.get(self.model_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                log_and_raise(ModelRegistryError, "Server did not provide Content-Length header")

            downloaded_size = 0

            with open(cached_file, "wb") as f:
                # Create a progress bar with unit='B' (bytes) and unit_scale=True to show sizes in MB/GB
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {filename}") as pbar:
                    for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            pbar.update(len(chunk))

            # Verify the downloaded file matches expected size
            if not self._verify_downloaded_file(cached_file, total_size):
                log_and_raise(ModelRegistryError, "Downloaded file is invalid: %s", cached_file)

            logger.info("Model downloaded successfully to %s", cached_file)
            return str(cached_file)
        except requests.exceptions.RequestException as e:
            # In case of a request error, provide a hint about the API key to assist users with authentication issues
            log_and_raise(
                ModelRegistryError,
                "Failed to download model: %s\nAPI key: '%s'\nAPI key hint: %s",
                str(e),
                # Don't log full API key for security reasons
                partial_api_key(unpack_optional(self.api_key), 8),
                self.api_key_hint(),
            )
        except Exception as e:
            log_and_raise(ModelRegistryError, "Unexpected error while downloading model: %s", str(e))

    def get_model(self) -> str:
        """Download a model or use cached version if available.

        This is the main method that should be called by clients to get a model.

        Returns:
            str: Path to the downloaded model file.

        Raises:
            ModelRegistryError: If the download fails.
        """
        # Create cache directory if it doesn't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured cache directory exists: %s", self.model_cache_dir)

        # Parse the model URL to get the filename
        parsed_url = urlparse(self.model_url)
        filename = os.path.basename(parsed_url.path)
        cached_file = self.model_cache_dir / filename
        logger.debug("Target cache file: %s", cached_file)

        # Check for existing file in cache
        if cached_file.exists():
            logger.debug("Found existing file in cache")
            if self._verify_cached_file(cached_file):
                logger.info("Using cached model file: %s", cached_file)
                return str(cached_file)
            else:
                logger.warning("Cached file is invalid, will redownload: %s", cached_file)
                cached_file.unlink()

        # If we got here, we need to download the file
        return self._download_to_file(filename)

    @staticmethod
    def _get_credential_from_netrc(domain: str) -> Optional[str]:
        """Get credential from the .netrc file for a specific domain.

        Args:
            domain (str): The domain to get credentials for.

        Returns:
            str: Credential if found in .netrc, None otherwise.

        Raises:
            ModelRegistryError: If there's an error reading the .netrc file.
        """
        logger.debug(f"Attempting to get credentials from the .netrc file for {domain}.")
        try:
            auth = netrc.netrc().authenticators(domain)
            if not auth:
                logger.debug("No authentication found in .netrc")
                return None
            credential = auth[2]
            if not credential:
                logger.debug("No credential found in .netrc")
                return None
            logger.debug(f"Successfully retrieved credential from the .netrc file for {domain}.")
            return credential
        except FileNotFoundError as e:
            logger.error("Failed to read .netrc file: %s", str(e))
            log_and_raise(ModelRegistryError, "Failed to read .netrc file: %s", str(e))


class NgcModelRegistry(ModelRegistry):
    """Implementation for downloading models from NVIDIA's NGC registry.

    This class handles downloading models from NVIDIA's NGC registry. It supports
    caching downloaded models to avoid redundant downloads.

    Attributes:
        api_key (str): API key for authenticating with NGC. If not provided, it will be
            retrieved from the NGC_API_KEY environment variable or from the .netrc file.
    """

    NGC_DOMAIN = "api.ngc.nvidia.com"
    API_KEY_HINT = (
        "NGC Personal Key with catalog rights (starting with 'nvapi-' - see https://ngc.nvidia.com/setup/api-keys)\n"
        + "NGC Legacy API key with catalog rights (not starting with 'nvapi-' - see https://ngc.nvidia.com/setup/api-keys)"
    )
    NGC_AUTH_DOMAIN = "authn.nvidia.com"

    @property
    def domain(self) -> str:
        return self.NGC_DOMAIN

    @staticmethod
    def api_key_hint() -> str:
        return NgcModelRegistry.API_KEY_HINT

    def _get_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """Get the API key for the NGC model registry.

        The API key is resolved in the following order of precedence:
        1. Directly provided api_key parameter
        2. NGC_API_KEY environment variable
        3. .netrc file credentials for the registry domain

        Args:
            api_key: Optional API key provided directly.

        Returns:
            Optional[str]: The API key for the NGC registry.

        Raises:
            ModelRegistryError: If no API key could be found or if an empty API key was provided.
        """
        logger.debug("Received api_key: %s.", api_key)
        # Validate API key if provided
        if api_key is not None and not api_key.strip():
            log_and_raise(ModelRegistryError, "API key cannot be empty")

        # Get API key from NGC_API_KEY environment variable if not provided
        if not api_key:
            logger.debug("No API key provided directly, attempting to get it from NGC_API_KEY environment variable.")
            api_key = os.environ.get("NGC_API_KEY")
            if api_key:
                logger.debug("Successfully retrieved API key from NGC_API_KEY environment variable.")

        # Get API key from .netrc file if not provided
        api_key = super()._get_api_key(api_key)

        # Check if api-key has the expected form of a NGC Personal Key and warn otherwise
        if api_key and (not api_key.startswith("nvapi-")):
            logger.warning(
                # Don't log full API key for security reasons
                f"The NGC API key '{partial_api_key(api_key, 8)}' does not appear to be a valid NGC Personal Key (expected to be 'nvapi-' prefixed) "
                "and might result in authentication issues [legacy NGC API keys are likely unsupported]."
            )

        return api_key

    def _extract_org_team_from_model_url(self) -> tuple[str, str]:
        """Extract org and team from self.model_url.

        Returns:
            tuple: (org, team) extracted from URL

        Raises:
            ModelRegistryError: If URL format is invalid
        """
        pattern = r"https://api\.ngc\.nvidia\.com/v2/org/([^/]+)/team/([^/]+)/.*"
        match = re.match(pattern, self.model_url)
        if not match:
            log_and_raise(ModelRegistryError, "Invalid NGC URL format, cannot extract org/team: %s", self.model_url)

        org, team = match.groups()
        logger.debug("Extracted org='%s', team='%s' from URL", org, team)
        return org, team

    def _request_session_token_from_legacy_api_key(self, legacy_api_key: str, org: str, team: str) -> str:
        """Exchange legacy API key for session token via NGC authentication service.

        Args:
            legacy_api_key: The legacy NGC API key
            org: Organization name (e.g., 'nvstaging')
            team: Team name (e.g., 'nurec')

        Returns:
            str: Session token for authenticated requests

        Raises:
            ModelRegistryError: If token exchange fails
        """
        logger.debug("Exchanging legacy API key for session token")

        # Construct the scope parameter for the authentication request
        scope = f"group/ngc:{org}&group/ngc:{org}/{team}"
        auth_url = f"https://{self.NGC_AUTH_DOMAIN}/token?service=ngc&scope={scope}"

        try:
            # Make the token exchange request
            response = requests.get(
                auth_url, auth=("$oauthtoken", legacy_api_key), headers={"Accept": "application/json"}, timeout=30
            )
            response.raise_for_status()

            # Extract the token from the response
            token_data = response.json()
            session_token = token_data.get("token")

            if not session_token:
                log_and_raise(ModelRegistryError, "No token found in authentication response")

            logger.debug("Successfully exchanged legacy API key for session token")
            return session_token

        except requests.exceptions.RequestException as e:
            log_and_raise(ModelRegistryError, "Failed to exchange legacy API key for session token: %s", str(e))
        except (KeyError, ValueError) as e:
            log_and_raise(ModelRegistryError, "Invalid response format from NGC authentication service: %s", str(e))

    def _request_session_token_from_api_key(self, api_key: str) -> str:
        """Get a session token for the NGC model registry with proper authentication.

        This method handles the NGC-specific authentication flow:
        1. Gets the legacy API key
        2. Extracts org/team from the model URL
        3. Exchanges legacy API key for session token

        Returns:
            session_token: A session token configured for NGC authentication
        """

        # Potential session token is the API key/personal access token (PAT)
        potential_session_token = super()._request_session_token_from_api_key(api_key)

        # PATs start with "nvapi-", legacy access tokens don't.
        if potential_session_token.startswith("nvapi-"):
            # PAT is the session token
            return potential_session_token
        else:
            # Legacy access tokens need to request session tokens for a given org and team
            org, team = self._extract_org_team_from_model_url()
            return self._request_session_token_from_legacy_api_key(potential_session_token, org, team)


class GitLabModelRegistry(ModelRegistry):
    """Implementation for downloading models from a GitLab instance.

    This class handles downloading models from a GitLab instance. It supports
    caching downloaded models to avoid redundant downloads. The GitLab host is
    configurable, so self-hosted instances can be used (see `register_gitlab_domain`
    and the NUREC_GITLAB_DOMAINS environment variable for the factory).

    Attributes:
        api_key (str): API key for authenticating with GitLab. If not provided, it will be
            retrieved from the .netrc file.
        domain (str): Host name of the GitLab instance. Defaults to gitlab.com.
    """

    DEFAULT_GITLAB_DOMAIN = "gitlab.com"

    def __init__(
        self,
        model_url: str,
        model_cache_dir: Path,
        api_key: Optional[str] = None,
        domain: str = DEFAULT_GITLAB_DOMAIN,
    ) -> None:
        # The base constructor validates the URL against self.domain, so set it first.
        self._domain = domain
        super().__init__(model_url=model_url, model_cache_dir=model_cache_dir, api_key=api_key)

    @property
    def domain(self) -> str:
        return self._domain

    def api_key_hint(self) -> str:
        return f"GitLab Personal Access Token with read rights (see https://{self.domain}/-/user_settings/personal_access_tokens)"


class HuggingFaceModelRegistry(ModelRegistry):
    """Implementation for downloading model files from Hugging Face Hub."""

    HUGGING_FACE_DOMAIN = "huggingface.co"
    API_KEY_HINT = "Hugging Face token with repository access, provided directly or via HF_TOKEN"

    @property
    def domain(self) -> str:
        return self.HUGGING_FACE_DOMAIN

    @staticmethod
    def api_key_hint() -> str:
        return HuggingFaceModelRegistry.API_KEY_HINT

    def _get_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """Get a Hugging Face token if one is available.

        Public Hugging Face repositories do not require a token, so None is a valid result.
        """
        if api_key is not None and not api_key.strip():
            log_and_raise(ModelRegistryError, "API key cannot be empty")
        if api_key:
            return api_key

        token = os.environ.get("HF_TOKEN")
        if token and token.strip():
            logger.debug("Successfully retrieved Hugging Face token from HF_TOKEN environment variable.")
            return token
        return None

    def _parse_model_url(self) -> tuple[str, str, str]:
        """Parse a Hugging Face file URL into repo id, revision and repo-relative filename.

        Accepts both single-segment repo IDs (e.g. .../gpt2/resolve/main/file)
        and two-segment org/repo IDs (e.g. .../org/repo/resolve/main/file).
        """
        parsed = urlparse(self.model_url)
        path_parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(path_parts) < 4:
            log_and_raise(ModelRegistryError, "Invalid Hugging Face URL format: %s", self.model_url)

        # Detect single-segment (path_parts[1] is resolve/blob) vs two-segment layout.
        if path_parts[1] in ("resolve", "blob"):
            repo_id = path_parts[0]
            path_type, revision = path_parts[1:3]
            filename = "/".join(path_parts[3:])
        else:
            if len(path_parts) < 5:
                log_and_raise(ModelRegistryError, "Invalid Hugging Face URL format: %s", self.model_url)
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            path_type, revision = path_parts[2:4]
            filename = "/".join(path_parts[4:])

        if path_type not in ("resolve", "blob"):
            log_and_raise(
                ModelRegistryError,
                "Unsupported Hugging Face URL type '%s' (expected 'resolve' or 'blob'): %s",
                path_type,
                self.model_url,
            )
        if not repo_id or not revision or not filename:
            log_and_raise(ModelRegistryError, "Invalid Hugging Face URL format: %s", self.model_url)

        return repo_id, revision, filename

    def _download_to_file(self, filename: str) -> str:
        """Download a single file from Hugging Face Hub into the configured cache dir.

        Hugging Face uses repo-relative filenames for downloads, but get_model()
        checks the flat <cache_dir>/<filename> path. Keep the final cached path
        aligned with that base ModelRegistry behavior.
        """
        logger.info("Starting model download from %s", self.__class__.__name__)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import (
                EntryNotFoundError,
                HfHubHTTPError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            )
        except ImportError as e:
            raise ModelRegistryError("huggingface-hub is required for Hugging Face model URLs") from e

        try:
            repo_id, revision, parsed_filename = self._parse_model_url()
            cached_file = self.model_cache_dir / filename
            cached_file.parent.mkdir(parents=True, exist_ok=True)

            self.api_key = self._get_api_key(self.api_key)

            logger.info("Downloading model from %s", self.model_url)
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=parsed_filename,
                revision=revision,
                local_dir=self.model_cache_dir,
                token=self.api_key,
            )
            downloaded_path = Path(downloaded_file)
            if not downloaded_path.exists() or not downloaded_path.is_file():
                log_and_raise(ModelRegistryError, "Downloaded file does not exist: %s", downloaded_file)
            if not self._verify_cached_file(downloaded_path):
                log_and_raise(ModelRegistryError, "Downloaded file is invalid: %s", downloaded_file)

            if downloaded_path.resolve() != cached_file.resolve():
                if cached_file.exists():
                    cached_file.unlink()
                shutil.move(str(downloaded_path), str(cached_file))

            logger.info("Model downloaded successfully to %s", cached_file)
            return str(cached_file)
        except (EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError) as e:
            log_and_raise(
                ModelRegistryError,
                "Failed to download Hugging Face model: %s\nAPI key hint: %s",
                str(e),
                self.api_key_hint(),
            )
        except ModelRegistryError:
            raise
        except Exception as e:
            log_and_raise(ModelRegistryError, "Unexpected error while downloading Hugging Face model: %s", str(e))


class ResolvedBundle(NamedTuple):
    """A downloaded Hugging Face model bundle and its parsed config.json manifest, if any."""

    path: Path
    manifest: Optional[Mapping[str, Any]]


def fetch_bundle(repo_id: str, revision: str, cache_dir: str, allow_patterns: Optional[list] = None) -> ResolvedBundle:
    """Fetch a multi-file Hugging Face model bundle and its config.json manifest, if present.

    Unlike ModelRegistry.get_model(), which downloads one named file and enforces a minimum
    file-size check meant for full model checkpoints, this downloads an entire Hugging Face
    snapshot (optionally filtered by allow_patterns) with no size constraint, so small metadata
    files such as config.json download successfully alongside checkpoints.

    Args:
        repo_id: Hugging Face repo id, e.g. "nvidia/asset-harvester".
        revision: Hugging Face revision (branch, tag, or commit sha).
        cache_dir: Directory where the snapshot should be cached.
        allow_patterns: Optional glob patterns to scope which files are downloaded.

    Returns:
        ResolvedBundle: The snapshot directory and its parsed config.json manifest (None if absent).

    Raises:
        ModelRegistryError: If the download fails.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as e:
        raise ModelRegistryError("huggingface-hub is required to fetch Hugging Face model bundles") from e

    try:
        snapshot_dir = Path(
            snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=allow_patterns)
        )
    except HfHubHTTPError as e:
        log_and_raise(ModelRegistryError, "Failed to download Hugging Face bundle %s@%s: %s", repo_id, revision, e)

    manifest_path = snapshot_dir / "config.json"
    if not manifest_path.exists():
        manifest = None
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log_and_raise(
                ModelRegistryError,
                "Invalid Hugging Face bundle manifest %s for %s@%s: %s",
                manifest_path,
                repo_id,
                revision,
                e,
            )
    return ResolvedBundle(path=snapshot_dir, manifest=manifest)


# Extra GitLab hosts registered at runtime via register_gitlab_domain().
_REGISTERED_GITLAB_DOMAINS: set[str] = set()

# Environment variable with comma-separated GitLab hosts, read on every factory call.
GITLAB_DOMAINS_ENV_VAR = "NUREC_GITLAB_DOMAINS"


def register_gitlab_domain(domain: str) -> None:
    """Register a GitLab host so create_model_registry maps it to GitLabModelRegistry.

    Registering the same domain more than once is a no-op.

    Args:
        domain: Bare host name (optionally with port), e.g. "gitlab.example.com".
            Must not include a scheme, path, or whitespace.

    Raises:
        ValueError: If the domain is empty or contains a scheme, path, or whitespace.
    """
    if not isinstance(domain, str):
        raise ValueError(f"GitLab domain must be a string, got {type(domain).__name__}")
    normalized = domain.strip().lower()
    if not normalized:
        raise ValueError("GitLab domain cannot be empty")
    if "://" in normalized or "/" in normalized or any(c.isspace() for c in normalized):
        raise ValueError(f"GitLab domain must be a bare host name without scheme, path, or whitespace: {domain!r}")
    _REGISTERED_GITLAB_DOMAINS.add(normalized)


def _get_gitlab_domains() -> set[str]:
    """Collect the known GitLab hosts: default, registered, and NUREC_GITLAB_DOMAINS."""
    domains = {GitLabModelRegistry.DEFAULT_GITLAB_DOMAIN}
    domains.update(_REGISTERED_GITLAB_DOMAINS)
    for entry in os.environ.get(GITLAB_DOMAINS_ENV_VAR, "").split(","):
        entry = entry.strip().lower()
        if entry:
            domains.add(entry)
    return domains


def create_model_registry(model_url: str, model_cache_dir: Path, api_key: Optional[str] = None) -> ModelRegistry:
    """Factory function to create the appropriate model registry based on the URL.

    The registry is selected by the URL's host. NGC, Hugging Face and gitlab.com are
    built in. Additional GitLab hosts can be enabled with `register_gitlab_domain` or
    by setting the NUREC_GITLAB_DOMAINS environment variable to a comma-separated list
    of hosts (read on each call; blank entries are ignored).

    Args:
        model_url: URL where the model can be downloaded from.
        model_cache_dir: Directory where downloaded models should be cached.
        api_key: Optional API key to authenticate with the registry.
            For NgcModelRegistry: If not provided, the API key will be retrieved from
            the NGC_API_KEY environment variable or from the .netrc file.
            For other registries: If not provided, the API key will be retrieved from
            the .netrc file.

    Returns:
        ModelRegistry: An instance of the appropriate model registry.

    Raises:
        ModelRegistryError: If the URL does not match any supported registry.
    """
    parsed_url = urlparse(model_url)
    domain = parsed_url.netloc

    logger.debug("Determining registry type for domain: %s", domain)

    if domain == NgcModelRegistry.NGC_DOMAIN:
        logger.debug("Creating NgcModelRegistry for domain: %s", domain)
        return NgcModelRegistry(model_url=model_url, model_cache_dir=model_cache_dir, api_key=api_key)

    if domain == HuggingFaceModelRegistry.HUGGING_FACE_DOMAIN:
        logger.debug("Creating HuggingFaceModelRegistry for domain: %s", domain)
        return HuggingFaceModelRegistry(model_url=model_url, model_cache_dir=model_cache_dir, api_key=api_key)

    if domain.lower() in _get_gitlab_domains():
        logger.debug("Creating GitLabModelRegistry for domain: %s", domain)
        return GitLabModelRegistry(model_url=model_url, model_cache_dir=model_cache_dir, api_key=api_key, domain=domain)

    # No matching registry found
    log_and_raise(ModelRegistryError, "Unsupported model registry domain: %s", domain)
