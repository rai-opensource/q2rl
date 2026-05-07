from backend.agents.cql_image import CQLImageAgent


class CalQLImageAgent(CQLImageAgent):
    """Same agent as CQL, just add an additional check that the use_calql flag is on."""

    @classmethod
    def create(
        cls,
        *args,
        **kwargs,
    ):
        kwargs["use_calql"] = True
        return super(CalQLImageAgent, cls).create(
            *args,
            **kwargs,
        )