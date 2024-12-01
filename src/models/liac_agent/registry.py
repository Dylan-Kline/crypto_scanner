from mmengine.registry import Registry

DATASET = Registry('data', locations=['liac_agent.data'])
PROMPT = Registry('prompt', locations=['liac_agent.prompt'])
AGENT = Registry('agent', locations=['liac_agent.agent'])
PROVIDER = Registry('provider', locations=['liac_agent.provider'])
DOWNLOADER = Registry('downloader', locations=['liac_agent.downloader'])
PROCESSOR = Registry('processor', locations=['liac_agent.processor'])
ENVIRONMENT = Registry('environment', locations=['liac_agent.environment'])
MEMORY = Registry('memory', locations=['liac_agent.memory'])
PLOTS = Registry('plot', locations=['liac_agent.plots'])