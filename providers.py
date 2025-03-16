cooldowns = {}
class Model:
    def __init__(self, base_url):
        self.base_url = base_url

class CooldownProvider:
    def __init__(self, cooldown_seconds):
        self.cooldown_seconds = cooldown_seconds

    def reset(self):
        pass

models = [Model("http://example1.com"), Model("http://example2.com")]
cooldown_seconds = 60
cooldowns = {base_url: CooldownProvider(cooldown_seconds) for base_url in set(model.base_url for model in models)}

def set_cooldown(base_url):
    if base_url not in cooldowns:
        cooldowns[base_url] = CooldownProvider(cooldown_seconds)
    else:
        cooldowns[base_url].reset()
