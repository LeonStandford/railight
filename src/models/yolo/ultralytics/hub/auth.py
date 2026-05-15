from __future__ import annotations
from ultralytics.hub.utils import (
    HUB_API_ROOT,
    HUB_WEB_ROOT,
    PREFIX,
    request_with_credentials,
)
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, emojis

API_KEY_URL = f"{HUB_WEB_ROOT}/settings?tab=api+keys"


class Auth:
    id_token = api_key = model_key = False

    def __init__(self, api_key: str = "", verbose: bool = False):
        api_key = api_key.split("_", 1)[0]
        self.api_key = api_key or SETTINGS.get("api_key", "")
        if self.api_key:
            if self.api_key == SETTINGS.get("api_key"):
                if verbose:
                    LOGGER.info(f"{PREFIX}Authenticated ✅")
                return
            else:
                success = self.authenticate()
        elif IS_COLAB:
            success = self.auth_with_cookies()
        else:
            success = self.request_api_key()
        if success:
            SETTINGS.update({"api_key": self.api_key})
            if verbose:
                LOGGER.info(f"{PREFIX}New authentication successful ✅")
        elif verbose:
            LOGGER.info(
                f"{PREFIX}Get API key from {API_KEY_URL} and then run 'yolo login API_KEY'"
            )

    def request_api_key(self, max_attempts: int = 3) -> bool:
        import getpass

        for attempts in range(max_attempts):
            LOGGER.info(f"{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}")
            input_key = getpass.getpass(f"Enter API key from {API_KEY_URL} ")
            self.api_key = input_key.split("_", 1)[0]
            if self.authenticate():
                return True
        raise ConnectionError(emojis(f"{PREFIX}Failed to authenticate ❌"))

    def authenticate(self) -> bool:
        import requests

        try:
            if header := self.get_auth_header():
                r = requests.post(f"{HUB_API_ROOT}/v1/auth", headers=header)
                if not r.json().get("success", False):
                    raise ConnectionError("Unable to authenticate.")
                return True
            raise ConnectionError("User has not authenticated locally.")
        except ConnectionError:
            self.id_token = self.api_key = False
            LOGGER.warning(f"{PREFIX}Invalid API key")
            return False

    def auth_with_cookies(self) -> bool:
        if not IS_COLAB:
            return False
        try:
            authn = request_with_credentials(f"{HUB_API_ROOT}/v1/auth/auto")
            if authn.get("success", False):
                self.id_token = authn.get("data", {}).get("idToken", None)
                self.authenticate()
                return True
            raise ConnectionError("Unable to fetch browser authentication details.")
        except ConnectionError:
            self.id_token = False
            return False

    def get_auth_header(self):
        if self.id_token:
            return {"authorization": f"Bearer {self.id_token}"}
        elif self.api_key:
            return {"x-api-key": self.api_key}
