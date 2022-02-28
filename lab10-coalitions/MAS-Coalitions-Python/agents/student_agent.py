from typing import List, Dict, Any

from agents import CoalitionAgent
from base import Coalition, Product
from communication import CoalitionAction


class MyCoalitionAgent(CoalitionAgent):
    PROD_TYPE_R1 = "r1"
    PROD_TYPE_R2 = "r2"

    def __init__(self, name: str, resources: float, products: List[Product]):
        super(MyCoalitionAgent, self).__init__(name, resources, products)

    def create_single_coalition(self) -> Coalition:
        c = Coalition(self.products)
        c.set_agent(self, share={self.PROD_TYPE_R1:
            {
                Coalition.PROD_CONTRIB: self.resources,
                Coalition.PROD_VALUE: 0
            }
        })
        return c

    def state_announced(self, agents: List[CoalitionAgent], coalitions: List[Coalition]) -> None:
        super().state_announced(agents, coalitions)

    def do_actions(self, messages: List[CoalitionAction] = None) -> List[CoalitionAction]:
        return []



