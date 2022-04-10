from tud_rl.agents.continuous import _CAGENTS
from tud_rl.agents.discrete import _DAGENTS
from tud_rl import logger

class AgentNotFoundError(Exception):
    pass

def validate_agent(agent: str) -> None:
    """Validates an agent_name string passed
    into it. 
    Returns an AgentNotFoundError 
    if the passed agent does not match
    any available agent, else None.
    """
    
    if agent not in _CAGENTS and agent not in _DAGENTS:
        logger.error(f"`{agent}` did not match any  of the "
                     "available agents.\nAvailable Agents:\n"
            f"{_CAGENTS}\n{_DAGENTS}"
        )
        raise AgentNotFoundError

def is_discrete(name: str) -> bool:
    return True if name in _DAGENTS else False

def is_continuous(name: str) -> bool:
    return True if name in _CAGENTS else False