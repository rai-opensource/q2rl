from .cql_image import CQLImageAgent
from .bc import BCAgent
from .calql import CalQLAgent
from .cql import CQLAgent
from .sac import SACAgent
from .q2rl_gym import Q2RLAgent
from .q2rl_robosuite import Q2RLAgent_RS
from .q2rl_robosuite_image import Q2RLAgent_RS_Image
from .ibrl_state import IBRLStateAgent
from .ibrl_image import IBRLImageAgent
from .sac_image import SACImageAgent
from .calql_image import CalQLImageAgent

agents = {
    "bc": BCAgent,
    "cql": CQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
    "q2rl": Q2RLAgent,
    "q2rl_rs": Q2RLAgent_RS,
    "q2rl_rs_image": Q2RLAgent_RS_Image,
    "ibrl_state": IBRLStateAgent,
    "ibrl_image": IBRLImageAgent,
    "sac_image": SACImageAgent,
    "cql_image": CQLImageAgent,
}
