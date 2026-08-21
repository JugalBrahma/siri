import unittest
from tools.human_tool import ask_human
from agents.actionagent import tools as action_tools
from agents.research_agent import tools as research_tools
from agents.weather_agent import weather_tools
from graph_builder.graph import GraphBuilder


class TestHumanInTheLoopTool(unittest.TestCase):
    def test_ask_human_tool_definition(self):
        self.assertEqual(ask_human.name, "ask_human")
        self.assertIn("clarification", ask_human.description.lower())

    def test_ask_human_present_in_action_agent(self):
        tool_names = [t.name for t in action_tools]
        self.assertIn("ask_human", tool_names)

    def test_ask_human_present_in_research_agent(self):
        tool_names = [t.name for t in research_tools]
        self.assertIn("ask_human", tool_names)

    def test_ask_human_present_in_weather_agent(self):
        tool_names = [t.name for t in weather_tools]
        self.assertIn("ask_human", tool_names)

    def test_graph_builder_has_checkpointer(self):
        builder = GraphBuilder(
            supervisor=lambda s: s,
            sub_infosupervisor=lambda s: s,
            sub_actionsupervisor=lambda s: s,
            researcher=lambda s: s,
            weather=lambda s: s,
            action=lambda s: s,
            guardrail=lambda s: s,
            output_sanitizer=lambda s: s,
        )
        self.assertIsNotNone(builder.checkpointer)
        compiled = builder.build()
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
