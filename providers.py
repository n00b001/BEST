cooldowns = {}
cooldowns = {base_url: CooldownProvider(cooldown_seconds) for model in models}

def set_cooldown(base_url):
    if base_url not in cooldowns:
        cooldowns[base_url] = CooldownProvider(cooldown_seconds)
