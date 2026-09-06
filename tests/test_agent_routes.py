from Models.schema import RouterSchema
from agents.data_agent import route_decision


def test_router_accepts_new_agent_routes():
    for route in [
        "sql",
        "etl",
        "visualization",
        "catalog",
        "quality",
        "lineage",
        "forecast",
        "security",
        "clarify",
    ]:
        parsed = RouterSchema(answer=route, comments="routing check")
        assert parsed.answer == route


def test_route_decision_handles_extended_agents():
    assert route_decision(type("State", (), {"route_response": "quality"})()) == "quality"
    assert route_decision(type("State", (), {"route_response": "lineage"})()) == "lineage"
    assert route_decision(type("State", (), {"route_response": "forecast"})()) == "forecast"
    assert route_decision(type("State", (), {"route_response": "security"})()) == "security"
