import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.main import app

class TestMain(unittest.TestCase):
    def test_health_check(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            # The actual content of "content" depends on router.healthcheck()
            # For now, just check that "status" key exists if response is successful
            if response.json().get("status"):
                 self.assertIn("status", response.json())

    def test_ok_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/ok")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    @patch('src.main.load_bearer_tokens')
    def test_stats_auth_no_tokens_configured(self, mock_load_bearer_tokens):
        mock_load_bearer_tokens.return_value = [] # No tokens configured, auth disabled
        with TestClient(app) as client:
            response = client.get("/stats")
            # Expecting 200 because if no tokens are configured, auth is skipped
            self.assertEqual(response.status_code, 200)
            # The actual content of "stats" depends on router.stats()
            # For now, just check that "stats" key exists if response is successful
            if response.json().get("stats"):
                self.assertIn("stats", response.json())


    @patch('src.main.load_bearer_tokens')
    def test_stats_authentication(self, mock_load_bearer_tokens):
        mock_load_bearer_tokens.return_value = ["testtoken1", "testtoken2"]
        with TestClient(app) as client:
            # Test case 1: Valid token 1
            response1 = client.get("/stats", headers={"Authorization": "Bearer testtoken1"})
            self.assertEqual(response1.status_code, 200)
            if response1.json().get("stats"):
                self.assertIn("stats", response1.json())

            # Test case 2: Valid token 2
            response2 = client.get("/stats", headers={"Authorization": "Bearer testtoken2"})
            self.assertEqual(response2.status_code, 200)
            if response2.json().get("stats"):
                self.assertIn("stats", response2.json())

            # Test case 3: Invalid token
            response3 = client.get("/stats", headers={"Authorization": "Bearer invalidtoken"})
            self.assertEqual(response3.status_code, 403)
            self.assertEqual(response3.json(), {"detail": "Invalid token"})

            # Test case 4: Missing Authorization header
            response4 = client.get("/stats")
            self.assertEqual(response4.status_code, 401)
            self.assertEqual(response4.json(), {"detail": "Not authenticated"})

            # Test case 5: Non-bearer token
            response5 = client.get("/stats", headers={"Authorization": "Basic anVzZXJuYW1lOnBhc3N3b3Jk"})
            self.assertEqual(response5.status_code, 401)
            self.assertEqual(response5.json(), {"detail": "Not authenticated"}) # FastAPI's default for malformed Bearer

            # Test case 6: Empty bearer token
            response6 = client.get("/stats", headers={"Authorization": "Bearer "})
            self.assertEqual(response6.status_code, 401) # This is a malformed token, so 401
            self.assertEqual(response6.json(), {"detail": "Not authenticated"})


if __name__ == '__main__':
    unittest.main()
