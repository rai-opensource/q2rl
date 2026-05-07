from backend.agents.cql import CQLAgent


class CalQLAgent(CQLAgent):
    """Same agent as CQL, just add an additional check that the use_calql flag is on."""

    @classmethod
    def create(
        cls,
        *args,
        **kwargs,
    ):
        kwargs["use_calql"] = True
        return super(CalQLAgent, cls).create(
            *args,
            **kwargs,
        )
