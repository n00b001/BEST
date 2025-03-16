import unittest
import sys
sys.path.append('/workspace')
from providers import Model, CooldownProvider, set_cooldown, cooldowns

class TestCooldownPerBaseUrl(unittest.TestCase):
    def setUp(self):
        self.model1 = Model("http://example1.com")
        self.model2 = Model("http://example2.com")
        self.cooldown_seconds = 60

    def test_cooldown_per_base_url(self):
        set_cooldown(self.model1.base_url)
        set_cooldown(self.model1.base_url)
        set_cooldown(self.model2.base_url)

        set_cooldown(self.model1.base_url); set_cooldown(self.model1.base_url); set_cooldown(self.model2.base_url); self.assertEqual(len(cooldowns), 2)
        self.assertIsInstance(cooldowns[self.model1.base_url], CooldownProvider)
        self.assertIsInstance(cooldowns[self.model2.base_url], CooldownProvider)

if __name__ == '__main__':
    unittest.main()
